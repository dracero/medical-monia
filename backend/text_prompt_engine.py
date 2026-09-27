"""
Text Prompt Segmentation Engine using BiomedCLIP.

Uses BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) to perform
zero-shot semantic segmentation of histological structures based on a text prompt.

Pipeline:
1. Encode text prompt with BiomedCLIP text encoder
2. Split image into overlapping patches
3. Encode each patch with BiomedCLIP image encoder
4. Compute cosine similarity → activation heatmap
5. Adaptive threshold (Otsu) → binary mask
6. Morphological refinement → contour extraction → polygons + overlay PNG
"""

import numpy as np
from PIL import Image
import io
import base64
import torch
import torch.nn.functional as F
from skimage.measure import find_contours, approximate_polygon, label, regionprops
from skimage.morphology import opening, closing, disk, remove_small_objects
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter, zoom

# Spanish → English medical term dictionary for common histological structures
ES_EN_MEDICAL_DICT = {
    # Membranes & Barriers
    "membrana basal": "basement membrane",
    "membrana celular": "cell membrane",
    "membrana plasmática": "plasma membrane",
    "lámina basal": "basal lamina",
    # Cell types
    "núcleos": "cell nuclei",
    "nucleos": "cell nuclei",
    "núcleos celulares": "cell nuclei",
    "nucleos celulares": "cell nuclei",
    "células de leydig": "Leydig cells interstitial cells",
    "celulas de leydig": "Leydig cells interstitial cells",
    "células de sertoli": "Sertoli cells",
    "celulas de sertoli": "Sertoli cells",
    "espermatogonias": "spermatogonia germ cells",
    "espermatocitos": "spermatocytes",
    "espermátidas": "spermatids",
    "espermatozoides": "spermatozoa mature sperm",
    "células germinales": "germ cells",
    "celulas germinales": "germ cells",
    "linfocitos": "lymphocytes",
    "macrófagos": "macrophages",
    "macrofagos": "macrophages",
    "fibroblastos": "fibroblasts",
    "células endoteliales": "endothelial cells",
    "celulas endoteliales": "endothelial cells",
    # Tissue types
    "epitelio": "epithelium epithelial tissue",
    "epitelio germinal": "germinal epithelium",
    "epitelio germinativo": "germinal epithelium spermatogenesis",
    "estroma": "stroma connective tissue",
    "tejido conectivo": "connective tissue stroma",
    "tejido intersticial": "interstitial tissue",
    "intersticio": "interstitium interstitial space",
    # Structures
    "luz tubular": "tubular lumen lumen cavity",
    "lumen": "lumen tubular cavity",
    "luces": "lumen tubular cavities",
    "túbulos seminíferos": "seminiferous tubules",
    "tubulos seminiferos": "seminiferous tubules",
    "túbulo seminífero": "seminiferous tubule",
    "tubulo seminifero": "seminiferous tubule",
    "vasos sanguíneos": "blood vessels vasculature",
    "vasos sanguineos": "blood vessels vasculature",
    "capilares": "capillaries blood vessels",
    "arteriolas": "arterioles small arteries",
    "vénulas": "venules small veins",
    "venulas": "venules small veins",
    # Layers & Regions
    "capa muscular": "muscular layer smooth muscle",
    "túnica albugínea": "tunica albuginea fibrous capsule",
    "tunica albuginea": "tunica albuginea fibrous capsule",
    "miofibroblastos": "myofibroblasts peritubular cells",
    "lámina propia": "lamina propria",
    "lamina propia": "lamina propria",
    # Pathological
    "fibrosis": "fibrosis fibrotic tissue",
    "inflamación": "inflammation inflammatory infiltrate",
    "inflamacion": "inflammation inflammatory infiltrate",
    "necrosis": "necrosis necrotic tissue",
    "edema": "edema fluid accumulation",
    # Staining-related
    "hematoxilina": "hematoxylin basophilic dark purple nuclei",
    "eosina": "eosin eosinophilic pink cytoplasm",
    "citoplasma": "cytoplasm eosinophilic cellular content",
    "cromatina": "chromatin nuclear material",
    "colágeno": "collagen fibers connective tissue",
    "colageno": "collagen fibers connective tissue",
}


def translate_prompt(text: str) -> str:
    """
    Translates a Spanish histological prompt to English.
    Falls back to the original text if no translation found.
    """
    text_lower = text.strip().lower()

    # Direct dictionary match
    if text_lower in ES_EN_MEDICAL_DICT:
        return ES_EN_MEDICAL_DICT[text_lower]

    # Partial match: check if any key is contained in the text
    for es_term, en_term in ES_EN_MEDICAL_DICT.items():
        if es_term in text_lower:
            return en_term

    # Return original (might already be English or an unknown term)
    return text


class TextPromptSegmentationEngine:
    """
    BiomedCLIP-based text-guided segmentation for histological images.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.preprocess = None
        self.device = None
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy-load BiomedCLIP model on first use."""
        if self._loaded:
            return

        import open_clip

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load BiomedCLIP
        self.model, self.preprocess_train, self.preprocess = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )

        self.model.eval()
        self._loaded = True
        print(f"[TextPromptEngine] BiomedCLIP loaded on {self.device}")

    def load_image(self, image_source) -> Image.Image:
        """Load PIL Image from path, base64 string, or bytes."""
        if isinstance(image_source, str):
            if image_source.startswith("data:image"):
                base64_data = image_source.split(",")[1]
                img = Image.open(io.BytesIO(base64.b64decode(base64_data))).convert("RGB")
            else:
                img = Image.open(image_source).convert("RGB")
        elif isinstance(image_source, bytes):
            img = Image.open(io.BytesIO(image_source)).convert("RGB")
        elif isinstance(image_source, Image.Image):
            img = image_source.convert("RGB")
        else:
            raise ValueError("Unsupported image source")
        return img

    @torch.no_grad()
    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text prompt into a normalized embedding."""
        tokens = self.tokenizer([text]).to(self.device)
        text_features = self.model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)
        return text_features

    @torch.no_grad()
    def _encode_image_patch(self, patch_pil: Image.Image) -> torch.Tensor:
        """Encode a single image patch into a normalized embedding."""
        patch_tensor = self.preprocess(patch_pil).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(patch_tensor)
        image_features = F.normalize(image_features, dim=-1)
        return image_features

    def _generate_activation_map(
        self,
        img: Image.Image,
        text_embedding: torch.Tensor,
        patch_size: int = 112,
        stride: int = 56,
    ) -> np.ndarray:
        """
        Slide a window over the image, compute CLIP similarity per patch,
        and assemble an activation heatmap.
        """
        w, h = img.size
        arr = np.array(img)

        # Compute grid dimensions
        n_cols = max(1, (w - patch_size) // stride + 1)
        n_rows = max(1, (h - patch_size) // stride + 1)

        # Activation grid (low resolution)
        activation_grid = np.zeros((n_rows, n_cols), dtype=np.float32)

        for row in range(n_rows):
            for col in range(n_cols):
                y0 = row * stride
                x0 = col * stride
                y1 = min(y0 + patch_size, h)
                x1 = min(x0 + patch_size, w)

                # Ensure minimum patch size
                if (x1 - x0) < 32 or (y1 - y0) < 32:
                    continue

                patch = img.crop((x0, y0, x1, y1))
                patch_embedding = self._encode_image_patch(patch)

                # Cosine similarity (both already normalized)
                similarity = (text_embedding @ patch_embedding.T).item()
                activation_grid[row, col] = similarity

        # Upsample activation grid to full image resolution
        if n_rows > 1 and n_cols > 1:
            zoom_y = h / n_rows
            zoom_x = w / n_cols
            heatmap = zoom(activation_grid, (zoom_y, zoom_x), order=1)
        else:
            heatmap = np.full((h, w), activation_grid.mean(), dtype=np.float32)

        # Ensure exact size
        heatmap = heatmap[:h, :w]

        return heatmap

    def _heatmap_to_mask(
        self, heatmap: np.ndarray, sensitivity: float = 0.5
    ) -> np.ndarray:
        """
        Convert activation heatmap to binary mask using adaptive thresholding.
        sensitivity: 0.0 (strict, fewer pixels) to 1.0 (loose, more pixels)
        """
        # Normalize heatmap to [0, 1]
        hmin, hmax = heatmap.min(), heatmap.max()
        if hmax - hmin < 1e-6:
            return np.zeros_like(heatmap, dtype=bool)
        heatmap_norm = (heatmap - hmin) / (hmax - hmin)

        # Smooth the heatmap
        heatmap_smooth = gaussian_filter(heatmap_norm, sigma=8)

        # Adaptive threshold using Otsu, modulated by sensitivity
        try:
            otsu_thresh = threshold_otsu(heatmap_smooth)
        except ValueError:
            otsu_thresh = 0.5

        # Adjust threshold: lower sensitivity → higher threshold (stricter)
        adjusted_thresh = otsu_thresh * (1.5 - sensitivity)
        adjusted_thresh = max(0.05, min(0.95, adjusted_thresh))

        binary_mask = heatmap_smooth > adjusted_thresh

        # Morphological cleanup
        selem = disk(5)
        binary_mask = closing(binary_mask, selem)
        binary_mask = opening(binary_mask, selem)

        # Remove very small objects
        binary_mask = remove_small_objects(binary_mask, min_size=200)

        return binary_mask

    def _extract_polygons(
        self, mask: np.ndarray
    ) -> list[dict]:
        """Extract contour polygons from binary mask."""
        labeled_mask = label(mask)
        props = regionprops(labeled_mask)
        polygons = []

        for i, prop in enumerate(props):
            if prop.area < 100:
                continue
            sub_mask = (labeled_mask == prop.label).astype(float)
            contours = find_contours(sub_mask, 0.5)
            if not contours:
                continue

            # Take the largest contour
            contours.sort(key=lambda c: len(c), reverse=True)
            poly = np.column_stack([contours[0][:, 1], contours[0][:, 0]])
            poly_simplified = approximate_polygon(poly, tolerance=2.5)

            cy, cx = map(int, prop.centroid)
            polygons.append({
                "id": f"text_seg_{i}",
                "center": [int(cx), int(cy)],
                "area_px": int(prop.area),
                "polygon": poly_simplified.tolist(),
            })

        return polygons

    def _generate_overlay_png(
        self,
        heatmap: np.ndarray,
        mask: np.ndarray,
        h: int,
        w: int,
        color: tuple = (6, 182, 212),
    ) -> tuple[str, str]:
        """
        Generate two base64 PNGs:
        1. Heatmap overlay (gradient visualization)
        2. Segmentation mask overlay (solid color with alpha)
        """
        # --- Heatmap PNG ---
        hmin, hmax = heatmap.min(), heatmap.max()
        if hmax - hmin > 1e-6:
            heatmap_norm = (heatmap - hmin) / (hmax - hmin)
        else:
            heatmap_norm = np.zeros_like(heatmap)

        # Crop to image size
        hn = heatmap_norm[:h, :w]

        heatmap_rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # Vectorized colormap: blue → cyan → yellow → red
        r = np.zeros_like(hn, dtype=np.uint8)
        g = np.zeros_like(hn, dtype=np.uint8)
        b = np.zeros_like(hn, dtype=np.uint8)

        # Band 1: val < 0.25 → blue ramp
        m1 = hn < 0.25
        b[m1] = (hn[m1] * 4 * 180).astype(np.uint8)

        # Band 2: 0.25 <= val < 0.5 → cyan transition
        m2 = (hn >= 0.25) & (hn < 0.5)
        t2 = (hn[m2] - 0.25) * 4
        g[m2] = (t2 * 220).astype(np.uint8)
        b[m2] = ((1 - t2) * 180 + t2 * 212).astype(np.uint8)

        # Band 3: 0.5 <= val < 0.75 → yellow transition
        m3 = (hn >= 0.5) & (hn < 0.75)
        t3 = (hn[m3] - 0.5) * 4
        r[m3] = (t3 * 255).astype(np.uint8)
        g[m3] = ((1 - t3) * 220 + t3 * 200).astype(np.uint8)
        b[m3] = ((1 - t3) * 212).astype(np.uint8)

        # Band 4: val >= 0.75 → red
        m4 = hn >= 0.75
        t4 = (hn[m4] - 0.75) * 4
        r[m4] = 255
        g[m4] = ((1 - t4) * 200).astype(np.uint8)

        alpha = (hn * 160).astype(np.uint8)

        heatmap_rgba[:, :, 0] = r
        heatmap_rgba[:, :, 1] = g
        heatmap_rgba[:, :, 2] = b
        heatmap_rgba[:, :, 3] = alpha

        heatmap_pil = Image.fromarray(heatmap_rgba, mode="RGBA")
        buf_heatmap = io.BytesIO()
        heatmap_pil.save(buf_heatmap, format="PNG")
        heatmap_b64 = "data:image/png;base64," + base64.b64encode(buf_heatmap.getvalue()).decode("utf-8")

        # --- Mask PNG ---
        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        mask_cropped = mask[:h, :w]
        mask_rgba[mask_cropped] = [color[0], color[1], color[2], 150]

        # Add brighter edge
        from skimage.morphology import erosion
        inner = erosion(mask_cropped, disk(3))
        edge = mask_cropped & ~inner
        mask_rgba[edge] = [min(255, color[0] + 80), min(255, color[1] + 80), min(255, color[2] + 80), 220]

        mask_pil = Image.fromarray(mask_rgba, mode="RGBA")
        buf_mask = io.BytesIO()
        mask_pil.save(buf_mask, format="PNG")
        mask_b64 = "data:image/png;base64," + base64.b64encode(buf_mask.getvalue()).decode("utf-8")

        return heatmap_b64, mask_b64

    def segment(
        self,
        image_source,
        text_prompt: str,
        sensitivity: float = 0.5,
        calibration_um_per_px: float = 0.5,
        patch_size: int = 112,
        stride: int = 56,
    ) -> dict:
        """
        Main entry point: segment a histological image based on a text prompt.

        Args:
            image_source: path, base64 string, or PIL Image
            text_prompt: text describing the structure to segment (Spanish or English)
            sensitivity: 0.0 (strict) to 1.0 (loose)
            calibration_um_per_px: physical calibration
            patch_size: size of sliding window patches
            stride: step between patches

        Returns:
            dict with mask PNG, heatmap PNG, polygons, metrics
        """
        self._ensure_loaded()

        img = self.load_image(image_source)
        w_img, h_img = img.size

        # Translate prompt if in Spanish
        original_prompt = text_prompt.strip()
        en_prompt = translate_prompt(original_prompt)

        # Build enriched prompt for BiomedCLIP context
        clip_prompt = f"a histology microscopy image of {en_prompt}"

        print(f"[TextPromptEngine] Prompt: '{original_prompt}' → '{clip_prompt}'")
        print(f"[TextPromptEngine] Image: {w_img}x{h_img}, patch={patch_size}, stride={stride}")

        # 1. Encode text
        text_embedding = self._encode_text(clip_prompt)

        # 2. Generate activation heatmap
        heatmap = self._generate_activation_map(img, text_embedding, patch_size, stride)

        # 3. Convert to binary mask
        mask = self._heatmap_to_mask(heatmap, sensitivity)

        # 4. Extract polygons
        polygons = self._extract_polygons(mask)

        # 5. Generate overlay PNGs
        heatmap_b64, mask_b64 = self._generate_overlay_png(heatmap, mask, h_img, w_img)

        # 6. Compute metrics
        total_mask_px = int(np.sum(mask))
        total_image_px = h_img * w_img
        percentage = round((total_mask_px / total_image_px) * 100, 2) if total_image_px > 0 else 0
        area_um2 = round(total_mask_px * (calibration_um_per_px ** 2), 1)
        area_mm2 = round(area_um2 / 1e6, 4)

        # Confidence: mean activation in the segmented region
        hmin, hmax = heatmap.min(), heatmap.max()
        if hmax - hmin > 1e-6 and total_mask_px > 0:
            heatmap_norm = (heatmap - hmin) / (hmax - hmin)
            mean_activation = float(np.mean(heatmap_norm[mask[:heatmap_norm.shape[0], :heatmap_norm.shape[1]]]))
        else:
            mean_activation = 0.0

        return {
            "status": "success",
            "prompt_original": original_prompt,
            "prompt_translated": en_prompt,
            "prompt_clip": clip_prompt,
            "image_dimensions": {"width": w_img, "height": h_img},
            "heatmap_png": heatmap_b64,
            "mask_png": mask_b64,
            "polygons": polygons,
            "metrics": {
                "total_segmented_px": total_mask_px,
                "percentage_of_image": percentage,
                "area_um2": area_um2,
                "area_mm2": area_mm2,
                "mean_confidence": round(mean_activation, 3),
                "num_regions": len(polygons),
            },
            "calibration_um_per_px": calibration_um_per_px,
        }


# Singleton instance
text_prompt_engine = TextPromptSegmentationEngine()
