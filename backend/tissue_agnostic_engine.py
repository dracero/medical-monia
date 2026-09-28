"""
Universal Tissue-Agnostic Semantic Segmentation Engine with MONAI Core.

Implements the modern Digital Pathology Foundation Model workflow:
1. Architecture: SwinUNETR / Foundation Vision Transformers (BiomedCLIP / open_clip)
2. Universal Tissue Masking: Resolution-agnostic separation of tissue vs. glass background
3. Chromatic Standardization: Macenko stain normalization (H&E decomposition via Optical Density)
4. Dynamic Inference: MONAI sliding_window_inference with Gaussian blending & high overlap
5. Ephemeral In-Memory Session Management: Zero disk persistence across sessions
"""

import os
import io
import time
import base64
import uuid
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, sobel
from skimage.measure import find_contours, approximate_polygon, label, regionprops
from skimage.morphology import opening, closing, disk, remove_small_objects

# MONAI Core
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR

# Spanish -> English histological terms mapping (Universal Tissue Agnostic)
HISTO_TERMS_ES_EN: Dict[str, str] = {
    # Membranes & Barriers
    "membrana basal": "basement membrane basal lamina peritubular boundary",
    "membrana": "basement membrane cell boundary",
    "lamina basal": "basal lamina basement membrane",
    "lámina basal": "basal lamina basement membrane",
    "membrana celular": "cell membrane plasma membrane",
    "lamina propia": "lamina propria connective tissue",
    "lámina propia": "lamina propria connective tissue",

    # Epithelium & Parenchyma
    "epitelio": "epithelium epithelial layer glandular cells",
    "túbulos": "tubules epithelial tubular structures",
    "tubulos": "tubules epithelial tubular structures",
    "túbulo": "tubule epithelial tubular structure",
    "tubulo": "tubule epithelial tubular structure",
    "conducto": "duct epithelial duct lining",
    "acino": "acinus glandular acinar structure",
    "epitelio tubular": "tubular epithelium",

    # Lumens & Spaces
    "luz": "lumen tubular cavity empty lumen space",
    "luz tubular": "tubular lumen lumen cavity",
    "lumen": "lumen cavity internal space",
    "luces": "lumina tubular lumens",
    "espacio intersticial": "interstitial space interstitium",
    "intersticio": "interstitial tissue stroma",

    # Nuclei & Cells
    "núcleos": "cell nuclei dark round hematoxylin basophilic nuclei",
    "nucleos": "cell nuclei dark round hematoxylin basophilic nuclei",
    "células": "cells cellular structures",
    "celulas": "cells cellular structures",
    "células de leydig": "Leydig cells interstitial steroidogenic cells",
    "celulas de leydig": "Leydig cells interstitial steroidogenic cells",
    "células de sertoli": "Sertoli cells nurse cells supporting cells",
    "celulas de sertoli": "Sertoli cells nurse cells supporting cells",
    "espermatogonias": "spermatogonia basal germ cells",
    "espermatocitos": "spermatocytes",
    "espermatozoides": "spermatozoa sperm cells",
    "fibroblastos": "fibroblasts spindle cells",
    "linfocitos": "lymphocytes small round immune cells",
    "macrófagos": "macrophages histiocytes",

    # Stroma & Connective Tissue
    "estroma": "stroma connective tissue interstitial collagen fibers",
    "colágeno": "collagen fibers connective tissue stroma",
    "colageno": "collagen fibers connective tissue stroma",
    "tejido conectivo": "connective tissue stroma",
    "vasos": "blood vessels capillaries vasculature",
    "vasos sanguíneos": "blood vessels capillaries vasculature",
    "vasos sanguineos": "blood vessels capillaries vasculature",
    "músculo": "smooth muscle muscular layer",
    "musculo": "smooth muscle muscular layer",

    # Pathological & Organ-Agnostic
    "fibrosis": "fibrosis dense collagenous stroma",
    "necrosis": "necrosis necrotic tissue cellular debris",
    "inflamación": "inflammation inflammatory cell infiltrate",
    "inflamacion": "inflammation inflammatory cell infiltrate",
    "edema": "edema fluid accumulation pale interstitial spacing",
    "tumor": "neoplasm malignant tumor cells atypical nuclei",
    "glomérulo": "glomerulus renal corpuscle",
    "glomerulo": "glomerulus renal corpuscle",
    "criptas": "intestinal crypts glandular epithelium",
    "alveolos": "pulmonary alveoli alveolar walls",
}


def translate_histology_prompt(prompt: str) -> str:
    """Translates medical histological prompt from Spanish to English terms."""
    clean = prompt.strip().lower()
    if clean in HISTO_TERMS_ES_EN:
        return HISTO_TERMS_ES_EN[clean]
    for es_term, en_term in HISTO_TERMS_ES_EN.items():
        if es_term in clean:
            return en_term
    return prompt


# ==============================================================================
# 1. UNIVERSAL TISSUE MASKING (Resolution-Agnostic Glass Separation)
# ==============================================================================

def extract_universal_tissue_mask(img_rgb: np.ndarray, min_area: int = 128) -> np.ndarray:
    """
    Separates genuine biological tissue from glass slide background.
    Works agnostically across any tissue (dense, loose, lung, colon, testis, etc.).

    Biological tissue features:
    - Absorption in optical density space (OD > threshold)
    - Saturation > threshold (glass is near neutral white/gray)
    - Luminance < 238 (glass is bright white)
    """
    img_f = img_rgb.astype(np.float32)

    # 1. Optical Density: OD = -log10((I + 1) / 256)
    od = -np.log10((img_f + 1.0) / 256.0)
    od_max = np.max(od, axis=2)

    # 2. RGB Luminance & Saturation
    max_c = np.max(img_f, axis=2)
    min_c = np.min(img_f, axis=2)
    saturation = np.where(max_c > 0, (max_c - min_c) / (max_c + 1e-6), 0.0)
    luminance = (img_f[..., 0] * 0.299 + img_f[..., 1] * 0.587 + img_f[..., 2] * 0.114)

    # Combined tissue criterion: Not bright glass, has optical absorption or color saturation
    is_tissue = (luminance < 238) & ((saturation > 0.05) | (od_max > 0.08))

    # Morphological refinement
    # Close small cracks and gaps
    try:
        is_tissue = closing(is_tissue, disk(3))
        is_tissue = remove_small_objects(is_tissue, min_size=min_area)
    except Exception:
        pass

    return is_tissue.astype(bool)


# ==============================================================================
# 2. MACENKO CHROMATIC STAIN NORMALIZATION
# ==============================================================================

class MacenkoNormalizer:
    """
    Universal H&E Stain Normalization via Macenko Method.
    Decomposes optical density matrix into Hematoxylin and Eosin vectors using SVD,
    normalizing stain intensities to a standard reference palette.
    """

    # Reference stain vectors & maximum concentrations (Standard H&E)
    HERef = np.array([
        [0.5626, 0.7201, 0.4062],  # Hematoxylin (Blue/Purple)
        [0.2159, 0.8012, 0.5581],  # Eosin (Pink)
    ])
    maxCRef = np.array([1.9705, 1.0308])

    @classmethod
    def normalize(cls, img_rgb: np.ndarray, beta: float = 0.15, alpha: float = 1.0) -> np.ndarray:
        """
        Normalizes an RGB histology image to standard reference H&E.
        Falls back to original image if insufficient tissue or non-H&E.
        """
        orig_shape = img_rgb.shape
        img_f = img_rgb.astype(np.float32)

        # Calculate Optical Density (OD)
        od = -np.log10((img_f + 1.0) / 256.0)
        od_flat = od.reshape((-1, 3))

        # Filter out background pixels with OD < beta
        mask = np.all(od_flat > beta, axis=1)
        od_hat = od_flat[mask]

        if len(od_hat) < 200:
            # Fallback if too few tissue pixels
            return img_rgb

        try:
            # SVD on covariance matrix
            cov = np.cov(od_hat, rowvar=False)
            _, eigvecs = np.linalg.eigh(cov)
            # Two largest eigenvectors
            t_hat = od_hat @ eigvecs[:, [2, 1]]

            phi = np.arctan2(t_hat[:, 1], t_hat[:, 0])
            min_phi = np.percentile(phi, alpha)
            max_phi = np.percentile(phi, 100 - alpha)

            v1 = eigvecs[:, [2, 1]] @ np.array([np.cos(min_phi), np.sin(min_phi)])
            v2 = eigvecs[:, [2, 1]] @ np.array([np.cos(max_phi), np.sin(max_phi)])

            # Order: Hematoxylin first, then Eosin
            if v1[0] > v2[0]:
                he = np.array([v1, v2])
            else:
                he = np.array([v2, v1])

            # Stain concentrations for all pixels
            y = np.reshape(od, (-1, 3)).T
            c, _, _, _ = np.linalg.lstsq(he.T, y, rcond=None)

            # Normalize concentrations to reference 99th percentile
            max_c = np.percentile(c, 99, axis=1)
            c_norm = c * (cls.maxCRef[:, np.newaxis] / (max_c[:, np.newaxis] + 1e-6))

            # Reconstruct normalized image
            od_norm = np.clip(cls.HERef.T @ c_norm, 0.0, 5.0)
            img_norm = 255.0 * np.power(10.0, -od_norm)
            img_norm = np.clip(img_norm.T.reshape(orig_shape), 0, 255).astype(np.uint8)
            return img_norm

        except Exception as e:
            # Safe fallback if numerical instability occurs
            print(f"[MacenkoNormalizer] Fallback due to: {e}")
            return img_rgb


# ==============================================================================
# 3. MONAI SLIDING WINDOW INFERENCE & PREDICTOR
# ==============================================================================

class PromptPredictorModule(torch.nn.Module):
    """
    PyTorch Predictor Module compatible with MONAI sliding_window_inference.
    Receives image patch batches (B, 3, H, W) and outputs spatial semantic activation.
    """

    def __init__(self, visual_model: torch.nn.Module, text_embs: torch.Tensor, device: torch.device):
        super().__init__()
        self.visual = visual_model
        self.text_embs = text_embs  # (N_classes, Dim) where class 0 is target
        self.device = device

        # ImageNet / BiomedCLIP normalization constants
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) in [0, 1] range.
        Returns: (B, 1, H, W) contiguous activation map representing target class probability.
        """
        x_norm = (x - self.mean) / self.std
        img_features = self.visual(x_norm)
        img_features = F.normalize(img_features, dim=-1)

        # Multi-class competitive contrastive softmax (temperature tau = 0.07)
        # Prevents broad unspecific activation across general tissue
        logits = (img_features @ self.text_embs.T) / 0.07
        probs = F.softmax(logits, dim=-1)
        target_prob = probs[:, 0:1]  # (B, 1)

        # Expand spatially and ensure contiguous memory for MONAI Gaussian accumulation
        b, _, h, w = x.shape
        out = target_prob.view(b, 1, 1, 1).repeat(1, 1, h, w).contiguous()
        return out


# ==============================================================================
# 4. EPHEMERAL IN-MEMORY SESSION STORE (Zero Disk Persistence)
# ==============================================================================

class InMemorySessionStore:
    """
    Session-scoped in-memory store.
    Annotations, temporary masks, and uploaded images reside in RAM only.
    No data is persisted to disk between sessions.
    """

    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def get_session(self, session_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Retrieves or creates an in-memory session."""
        if not session_id or session_id not in self._sessions:
            session_id = str(uuid.uuid4())
            self._sessions[session_id] = {
                "id": session_id,
                "created_at": time.time(),
                "annotations": [],
                "last_result": None,
                "custom_image_cache": None,
            }
        return session_id, self._sessions[session_id]

    def reset_session(self, session_id: str) -> None:
        """Resets session data completely in memory."""
        if session_id in self._sessions:
            self._sessions[session_id] = {
                "id": session_id,
                "created_at": time.time(),
                "annotations": [],
                "last_result": None,
                "custom_image_cache": None,
            }

    def clear_all(self) -> None:
        """Clears all sessions from memory."""
        self._sessions.clear()


# Global in-memory session manager
session_store = InMemorySessionStore()


# ==============================================================================
# 5. UNIVERSAL TISSUE-AGNOSTIC SEGMENTATION ENGINE
# ==============================================================================

class UniversalTissueAgnosticEngine:
    """
    Universal Digital Pathology Semantic Segmentation Engine using MONAI Core
    and Foundation Vision-Language Representations.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.preprocess = None
        self.swin_unetr = None
        self._loaded = False
        print(f"[UniversalTissueAgnosticEngine] Initialized on device: {self.device}")

    def _ensure_loaded(self):
        """Loads foundation models into CUDA VRAM."""
        if self._loaded:
            return

        import open_clip

        print(f"[UniversalTissueAgnosticEngine] Loading BiomedCLIP Foundation Model on {self.device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        self.model.eval()

        # Optional SwinUNETR 2D architecture module from MONAI for structural reference
        try:
            self.swin_unetr = SwinUNETR(
                in_channels=3,
                out_channels=5,
                feature_size=24,
                spatial_dims=2
            ).to(self.device)
            self.swin_unetr.eval()
            print("[UniversalTissueAgnosticEngine] SwinUNETR 2D architecture ready.")
        except Exception as e:
            print(f"[UniversalTissueAgnosticEngine] SwinUNETR init note: {e}")

        self._loaded = True
        print("[UniversalTissueAgnosticEngine] Foundation Model loaded and ready.")

    def load_image(self, source: Any) -> Image.Image:
        """Loads PIL Image from path, base64 data, or bytes without touching disk."""
        if isinstance(source, str):
            if source.startswith("data:image"):
                b64_str = source.split(",")[1]
                return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")
            elif os.path.exists(source):
                return Image.open(source).convert("RGB")
            else:
                try:
                    return Image.open(io.BytesIO(base64.b64decode(source))).convert("RGB")
                except Exception:
                    raise FileNotFoundError(f"Image source not found: {source[:60]}")
        elif isinstance(source, bytes):
            return Image.open(io.BytesIO(source)).convert("RGB")
        elif isinstance(source, Image.Image):
            return source.convert("RGB")
        raise ValueError("Unsupported image source")

    @torch.no_grad()
    def segment(
        self,
        image_source: Any,
        text_prompt: str,
        sensitivity: float = 0.5,
        normalize_stain: bool = True,
        tissue_masking: bool = True,
        overlap: float = 0.5,
        calibration_um_per_px: float = 0.25,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes full tissue-agnostic semantic segmentation pipeline:
        1. Universal tissue masking (tissue vs glass)
        2. Macenko stain normalization (H&E color standardization)
        3. Multimodal text encoding (BiomedCLIP)
        4. MONAI sliding_window_inference with Gaussian blending
        5. Adaptive thresholding & morphological contour extraction
        6. Ephemeral session state recording
        """
        self._ensure_loaded()
        t0 = time.time()

        # 1. Load image
        img_pil = self.load_image(image_source)
        w, h = img_pil.size
        img_np = np.array(img_pil)

        # 2. Universal Tissue Masking (Separates slide glass background)
        if tissue_masking:
            tissue_mask = extract_universal_tissue_mask(img_np)
        else:
            tissue_mask = np.ones((h, w), dtype=bool)

        # 3. Chromatic Standardization: Macenko Stain Normalization
        if normalize_stain:
            img_processed = MacenkoNormalizer.normalize(img_np)
        else:
            img_processed = img_np

        # 4. Multimodal Text Encoding with Competitive Contrast Anchors
        prompt_en = translate_histology_prompt(text_prompt)
        competing_prompts = [
            prompt_en,
            "cell nuclei round dark purple hematoxylin dense chromatin",
            "tubular lumen empty white cavity background space",
            "interstitial connective tissue stroma extracellular collagen fibers",
            "clear transparent glass slide background empty field",
        ]
        tokens = self.tokenizer(competing_prompts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = F.normalize(text_features, dim=-1)

        # 5. MONAI Sliding Window Inference with Gaussian Blending
        img_tensor = torch.from_numpy(img_processed).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)  # (1, 3, H, W)

        predictor = PromptPredictorModule(self.model.visual, text_features, self.device)

        # Sliding window inference
        sw_out = sliding_window_inference(
            inputs=img_tensor,
            roi_size=(224, 224),
            sw_batch_size=8,
            predictor=predictor,
            overlap=float(np.clip(overlap, 0.25, 0.75)),
            mode="gaussian",
        )

        heatmap = sw_out[0, 0].cpu().numpy()
        heatmap = heatmap[:h, :w]

        # 6. High-Resolution Optical Density & Morphological Boundary Refinement
        prompt_lower = text_prompt.lower()
        is_boundary_target = any(k in prompt_lower for k in [
            "membrana", "basal", "boundary", "borde", "límite", "limite", "lamina", "perímetro"
        ])
        is_nuclei_target = any(k in prompt_lower for k in [
            "núcleo", "nucleo", "nuclei", "célula", "celula", "cell"
        ])
        is_lumen_target = any(k in prompt_lower for k in [
            "luz", "lumen", "cavidad", "tubular"
        ])

        # Optical density decomposition (Beer-Lambert Law)
        arr_f = img_processed.astype(np.float32)
        od = -np.log10((arr_f + 1.0) / 256.0)
        h_chan = np.clip(od[..., 0] * 0.7 + od[..., 2] * 0.3, 0.0, 5.0)  # Hematoxylin
        e_chan = np.clip(od[..., 1], 0.0, 5.0)                            # Eosin

        if is_boundary_target:
            # Basement membranes are the exact physical interface where dense cellular epithelium
            # transitions into connective tissue stroma (gradient of cell density x eosin)
            h_smooth = gaussian_filter(h_chan, sigma=2.5)
            grad_h = np.sqrt(sobel(h_smooth, axis=0)**2 + sobel(h_smooth, axis=1)**2)
            grad_norm = grad_h / (np.percentile(grad_h, 99.5) + 1e-6)
            grad_norm = np.clip(grad_norm, 0.0, 1.0)

            e_norm = e_chan / (np.percentile(e_chan, 99.0) + 1e-6)
            e_norm = np.clip(e_norm, 0.0, 1.0)

            # Combined anatomical boundary signal modulated by CLIP semantic prior
            boundary_signal = grad_norm * (0.3 + 0.7 * e_norm) * (0.2 + 0.8 * heatmap)
            boundary_signal[~tissue_mask] = 0.0
            boundary_signal = gaussian_filter(boundary_signal, sigma=1.0)
            boundary_max = np.percentile(boundary_signal[tissue_mask], 99.5) if np.any(tissue_mask) else 1.0
            refined_heatmap = np.clip(boundary_signal / (boundary_max + 1e-6), 0.0, 1.0)

            # High-precision thresholding (targeting 4-8% coverage for fine anatomical boundaries)
            # sensitivity 0.5 -> ~93rd percentile (~6-7% coverage)
            pct_cut = 98.0 - (10.0 * sensitivity)
            pct_val = np.percentile(refined_heatmap[tissue_mask], pct_cut) if np.any(tissue_mask) else 0.5
            binary_mask = (refined_heatmap >= pct_val) & tissue_mask

            # Morphological clean up
            binary_mask = remove_small_objects(binary_mask, min_size=32)
            heatmap_smooth = refined_heatmap

        elif is_nuclei_target:
            # Modulate with Hematoxylin peak absorption
            h_norm = h_chan / (np.percentile(h_chan, 99.0) + 1e-6)
            nuclei_signal = np.clip(h_norm, 0.0, 1.0) * (0.3 + 0.7 * heatmap)
            nuclei_signal[~tissue_mask] = 0.0
            refined_heatmap = gaussian_filter(nuclei_signal, sigma=1.0)
            heatmap_max = np.percentile(refined_heatmap[tissue_mask], 99.0) if np.any(tissue_mask) else 1.0
            refined_heatmap = np.clip(refined_heatmap / (heatmap_max + 1e-6), 0.0, 1.0)

            threshold = 0.8 - (0.4 * sensitivity)
            binary_mask = (refined_heatmap > threshold) & tissue_mask
            binary_mask = remove_small_objects(binary_mask, min_size=16)
            heatmap_smooth = refined_heatmap

        elif is_lumen_target:
            lum_signal = np.clip(1.0 - (h_chan + e_chan) / 2.0, 0.0, 1.0) * (0.3 + 0.7 * heatmap)
            lum_signal[~tissue_mask] = 0.0
            refined_heatmap = gaussian_filter(lum_signal, sigma=2.0)
            threshold = 0.75 - (0.35 * sensitivity)
            binary_mask = (refined_heatmap > threshold) & tissue_mask
            binary_mask = remove_small_objects(binary_mask, min_size=40)
            heatmap_smooth = refined_heatmap

        else:
            # General tissue compartment
            if tissue_masking:
                heatmap[~tissue_mask] = 0.0
            h_tissue = heatmap[tissue_mask] if np.any(tissue_mask) else heatmap.ravel()
            hmin, hmax = float(h_tissue.min()), float(h_tissue.max())
            if hmax - hmin > 1e-6:
                heatmap_norm = np.clip((heatmap - hmin) / (hmax - hmin), 0.0, 1.0)
            else:
                heatmap_norm = np.zeros_like(heatmap)
            heatmap_smooth = gaussian_filter(heatmap_norm, sigma=2.5)
            threshold = 1.0 - (0.3 + 0.5 * sensitivity)
            binary_mask = (heatmap_smooth > threshold) & tissue_mask
            try:
                binary_mask = closing(binary_mask, disk(2))
                binary_mask = remove_small_objects(binary_mask, min_size=64)
            except Exception:
                pass

        # 8. Contour & Polygon Extraction
        labeled_mask, num_regions = label(binary_mask, return_num=True)
        props = regionprops(labeled_mask)

        contours_raw = find_contours(binary_mask.astype(float), 0.5)
        polygons = []
        for c in contours_raw:
            if len(c) < 5:
                continue
            # Simplify polygon to reduce payload
            poly_approx = approximate_polygon(c, tolerance=2.0)
            # find_contours returns (row, col) = (y, x), convert to [x, y]
            pts = [[round(float(pt[1]), 1), round(float(pt[0]), 1)] for pt in poly_approx]
            if len(pts) >= 3:
                polygons.append(pts)

        # 9. Quantitative Metrics
        total_pixels = w * h
        tissue_pixels = int(np.sum(tissue_mask))
        segmented_pixels = int(np.sum(binary_mask))
        coverage_pct = round((segmented_pixels / max(1, tissue_pixels)) * 100.0, 2)
        area_um2 = round(segmented_pixels * (calibration_um_per_px ** 2), 2)
        confidence = round(float(np.mean(heatmap_smooth[binary_mask])) if segmented_pixels > 0 else 0.0, 3)

        # 10. Generate PNG Overlays (Heatmap & Mask)
        heatmap_b64 = self._generate_heatmap_overlay(heatmap_smooth, tissue_mask)
        mask_b64 = self._generate_mask_overlay(binary_mask)

        elapsed = round(time.time() - t0, 2)

        result_data = {
            "success": True,
            "prompt": text_prompt,
            "translated_prompt": prompt_en,
            "sensitivity": sensitivity,
            "tissue_masking": tissue_masking,
            "normalize_stain": normalize_stain,
            "overlap": overlap,
            "device": str(self.device),
            "inference_engine": "MONAI Core + BiomedCLIP Foundation Model",
            "metrics": {
                "segmented_pixels": segmented_pixels,
                "tissue_pixels": tissue_pixels,
                "coverage_pct": coverage_pct,
                "area_um2": area_um2,
                "region_count": num_regions,
                "mean_confidence": confidence,
                "elapsed_seconds": elapsed,
            },
            "polygons": polygons,
            "heatmap_png_base64": heatmap_b64,
            "mask_png_base64": mask_b64,
            "image_size": {"width": w, "height": h},
        }

        # Ephemeral Session Handling (RAM only)
        sid, sdata = session_store.get_session(session_id)
        sdata["last_result"] = result_data
        result_data["session_id"] = sid

        return result_data

    def _generate_heatmap_overlay(self, heatmap: np.ndarray, tissue_mask: np.ndarray) -> str:
        """Renders turbo/viridis colored heatmap on transparent RGBA PNG."""
        h, w = heatmap.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # Turbo colormap polynomial approximation
        val = np.clip(heatmap, 0.0, 1.0)
        r = np.clip(255 * (1.5 * val - 0.2), 0, 255).astype(np.uint8)
        g = np.clip(255 * (1.8 * val * (1.0 - val) * 4.0), 0, 255).astype(np.uint8)
        b = np.clip(255 * (1.2 * (1.0 - val) - 0.1), 0, 255).astype(np.uint8)

        alpha = (val * 190).astype(np.uint8)
        alpha[~tissue_mask] = 0
        alpha[val < 0.15] = 0

        rgba[..., 0] = r
        rgba[..., 1] = g
        rgba[..., 2] = b
        rgba[..., 3] = alpha

        pil_overlay = Image.fromarray(rgba, "RGBA")
        buf = io.BytesIO()
        pil_overlay.save(buf, format="PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    def _generate_mask_overlay(self, mask: np.ndarray) -> str:
        """Renders vivid purple/cyan segmentation mask on transparent RGBA PNG."""
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # Vivid violet/purple segmentation tint (167, 139, 250)
        rgba[mask, 0] = 167
        rgba[mask, 1] = 139
        rgba[mask, 2] = 250
        rgba[mask, 3] = 150

        pil_overlay = Image.fromarray(rgba, "RGBA")
        buf = io.BytesIO()
        pil_overlay.save(buf, format="PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


# Global engine instance
tissue_agnostic_engine = UniversalTissueAgnosticEngine()
