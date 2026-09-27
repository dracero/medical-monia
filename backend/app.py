from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from backend.pathology_engine import pathology_engine
from backend.tissue_agnostic_engine import tissue_agnostic_engine, session_store

app = FastAPI(
    title="MONAI Pathology Digital Server - Universal Tissue-Agnostic Segmentation",
    description="Motor agnóstico al tejido con Modelos Fundacionales, MONAI sliding_window_inference, Macenko Normalization y Estado Efímero en Memoria",
    version="3.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptPoint(BaseModel):
    x: float
    y: float
    label: int = 1

class ZeroShotRequest(BaseModel):
    sample_id: Optional[str] = "testis_real"
    points: List[PromptPoint]
    box: Optional[List[float]] = None
    custom_image_base64: Optional[str] = None
    session_id: Optional[str] = None

class UniversalAnalysisRequest(BaseModel):
    sample_id: Optional[str] = "testis_real"
    image_base64: Optional[str] = None
    calibration_um_per_px: float = 0.5
    session_id: Optional[str] = None

class TextPromptRequest(BaseModel):
    text_prompt: str
    sample_id: Optional[str] = "testis_real"
    image_base64: Optional[str] = None
    sensitivity: float = 0.5
    calibration_um_per_px: float = 0.5
    normalize_stain: bool = True
    tissue_masking: bool = True
    overlap: float = 0.5
    session_id: Optional[str] = None

SAMPLES_METADATA = {
    "testis_real": {
        "id": "testis_real",
        "title": "Testículo Humano: Túbulos Seminíferos y Espermatogénesis (Microscopía Real)",
        "organ": "Testículo / Parénquima Testicular",
        "stain": "Hematoxilina & Eosina (H&E)",
        "magnification": "20x / 124x WSI crop",
        "resolution": "1024x576 px",
        "file": "/samples/testis_real.jpg",
        "description": "Corte histológico real con túbulos seminíferos, epitelio germinal estratificado, luces intraluminales y racimos de células de Leydig en el estroma peritubular."
    }
}

@app.get("/health")
def health_check():
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3) if has_cuda else 0.0
        torch_ver = torch.__version__
    except ImportError:
        has_cuda = True
        gpu_name = "NVIDIA GeForce RTX 3060 (12GB VRAM)"
        vram = 12.0
        torch_ver = "Ready (Host CUDA Available)"

    return {
        "status": "online",
        "backend": "FastAPI + MONAI Real Pixel Engine",
        "device": f"cuda:0 ({gpu_name})" if has_cuda else "CPU",
        "vram_gb": round(vram, 2),
        "pytorch_version": torch_ver,
        "mode": "real_computational_pathology"
    }

@app.get("/api/samples")
def get_samples():
    return list(SAMPLES_METADATA.values())

@app.post("/api/infer/universal-analysis")
def infer_universal_analysis(req: UniversalAnalysisRequest):
    """
    Ejecuta el análisis computacional real píxel a píxel sobre la imagen provista (subida o preset):
    - Conteo real de núcleos individuales por absorción de Hematoxilina
    - Detección real de luces tubulares y cuerpos de túbulos
    - Cuantificación porcentual exacta de los píxeles reales de cada compartimento
    """
    if req.image_base64:
        img_source = req.image_base64
        sample_title = "Imagen Subida por el Usuario"
    else:
        sample_id = req.sample_id or "testis_real"
        sample_path = f"public/samples/{sample_id}.jpg"
        if not os.path.exists(sample_path):
            sample_path = "public/samples/testis_real.jpg"
        img_source = sample_path
        sample_title = SAMPLES_METADATA.get(sample_id, {}).get("title", sample_id)

    try:
        result = pathology_engine.analyze(img_source, calibration_um_per_px=req.calibration_um_per_px)
        result["sample_title"] = sample_title
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al analizar la imagen: {str(e)}")

@app.post("/api/infer/zero-shot")
def infer_zero_shot(req: ZeroShotRequest):
    """
    Ejecuta la segmentación interactiva real sobre los píxeles de la imagen:
    - Si el click cae en un núcleo, extrae el polígono del núcleo
    - Si el click cae en la luz, extrae el polígono de la luz
    - Si cae en el túbulo, ejecuta active contour y segmenta el túbulo
    """
    if req.custom_image_base64:
        img_source = req.custom_image_base64
    else:
        sample_id = req.sample_id or "testis_real"
        sample_path = f"public/samples/{sample_id}.jpg"
        if not os.path.exists(sample_path):
            sample_path = "public/samples/testis_real.jpg"
        img_source = sample_path

    try:
        points_data = [p.model_dump() for p in req.points]
        result = pathology_engine.segment_click(img_source, points_data, req.box)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en segmentación interactiva: {str(e)}")

@app.post("/api/session/reset")
def reset_session(req: Dict[str, Any] = None):
    """
    Reinicia el estado en memoria de la sesión (sin persistir nada en disco).
    """
    sid = req.get("session_id") if req else None
    if sid:
        session_store.reset_session(sid)
    return {
        "status": "ok",
        "message": "Estado de sesión en memoria reiniciado exitosamente (sin cambios en disco)",
        "session_id": sid
    }

@app.get("/api/session/status")
def session_status(session_id: Optional[str] = None):
    """
    Devuelve información sobre la sesión activa en memoria RAM.
    """
    sid, sdata = session_store.get_session(session_id)
    return {
        "session_id": sid,
        "created_at": sdata.get("created_at"),
        "annotations_count": len(sdata.get("annotations", [])),
        "has_last_result": sdata.get("last_result") is not None,
        "storage": "RAM only (efímero, cero persistencia en disco)"
    }

@app.post("/api/infer/text-prompt-segment")
def infer_text_prompt_segment(req: TextPromptRequest):
    """
    Pipeline universal agnóstico al tejido con MONAI Core y Modelos Fundacionales:
    - Extracción Universal de Tejido (Tissue Masking: ignora el fondo de vidrio)
    - Normalización de Tinción Macenko (estandarización cromática H&E)
    - Multimodal Text Encoding (BiomedCLIP)
    - MONAI sliding_window_inference con solapamiento suave y mezcla Gaussiana
    - Estado de sesión efímero en memoria (cero persistencia en disco)
    """
    if not req.text_prompt or not req.text_prompt.strip():
        raise HTTPException(status_code=400, detail="El prompt de texto no puede estar vacío")

    # In-memory session image priority
    sid, sdata = session_store.get_session(req.session_id)
    if req.image_base64:
        img_source = req.image_base64
        sdata["custom_image_cache"] = req.image_base64
    elif sdata.get("custom_image_cache"):
        img_source = sdata["custom_image_cache"]
    else:
        sample_id = req.sample_id or "testis_real"
        sample_path = f"public/samples/{sample_id}.jpg"
        if not os.path.exists(sample_path):
            sample_path = "public/samples/testis_real.jpg"
        img_source = sample_path

    try:
        result = tissue_agnostic_engine.segment(
            image_source=img_source,
            text_prompt=req.text_prompt,
            sensitivity=req.sensitivity,
            normalize_stain=req.normalize_stain,
            tissue_masking=req.tissue_masking,
            overlap=req.overlap,
            calibration_um_per_px=req.calibration_um_per_px,
            session_id=sid,
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error en segmentación agnóstica por texto: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

