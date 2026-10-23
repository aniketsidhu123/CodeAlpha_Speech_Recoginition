import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil

from src.config import (
    BASE_DIR,
    BEST_MODEL_PATH,
    DEMO_SAMPLES_DIR,
    EMOTION_CLASSES,
    EMOTION_COLORS
)
from src.predictor import EmotionPredictor

app = FastAPI(title="Speech Emotion Recognition API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file directories
STATIC_DIR = BASE_DIR / "web" / "static"
TEMPLATES_DIR = BASE_DIR / "web" / "templates"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/demo_audio", StaticFiles(directory=str(DEMO_SAMPLES_DIR)), name="demo_audio")

# Global predictor instance
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        if not BEST_MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model has not been trained yet. Please train the model using 'python main.py train'."
            )
        _predictor = EmotionPredictor()
    return _predictor

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Index template not found")
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))

@app.get("/api/status")
async def get_system_status():
    samples = [f.stem for f in DEMO_SAMPLES_DIR.glob("*.wav")]
    return {
        "model_trained": BEST_MODEL_PATH.exists(),
        "model_name": BEST_MODEL_PATH.name if BEST_MODEL_PATH.exists() else None,
        "emotions": EMOTION_CLASSES,
        "colors": EMOTION_COLORS,
        "demo_samples": samples
    }

@app.post("/api/predict/upload")
async def predict_uploaded_audio(file: UploadFile = File(...)):
    """Predict emotion from uploaded audio file or recorded voice note."""
    predictor = get_predictor()
    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
        result = predictor.predict_bytes(content)
        result["filename"] = file.filename
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/predict/sample/{sample_name}")
async def predict_demo_sample(sample_name: str):
    """Predict emotion from a pre-loaded RAVDESS demo sample."""
    predictor = get_predictor()
    sample_file = DEMO_SAMPLES_DIR / f"{sample_name.lower()}.wav"
    if not sample_file.exists():
        raise HTTPException(status_code=404, detail=f"Sample '{sample_name}' not found.")
        
    try:
        result = predictor.predict_file(sample_file)
        result["audio_url"] = f"/demo_audio/{sample_name.lower()}.wav"
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="127.0.0.1", port=8000, reload=True)
