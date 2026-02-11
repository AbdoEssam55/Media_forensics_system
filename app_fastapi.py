from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import tempfile
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from media_forensics_complete import (
    ImageDetector, AudioDetector, VideoDetector, Config
)


app = FastAPI(
    title="Media Forensics Detection API",
    description="Detect AI-generated images, audio, and videos",
    version="1.0.0"
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_detector = ImageDetector(device=device)
audio_detector = AudioDetector(device=device)
video_detector = VideoDetector(device=device)


class DetectionResult(BaseModel):
    classification: str
    confidence: float
    media_type: str
    has_audio: Optional[bool] = None
    visual_score: Optional[float] = None
    audio_score: Optional[float] = None
    temporal_consistency: Optional[float] = None


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": str(device),
        "cuda_available": torch.cuda.is_available()
    }


@app.post("/detect/image", response_model=DetectionResult)
async def detect_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name
    
    try:
        result = image_detector.detect(tmp_path)
        
        if result.get('status') == 'success':
            return DetectionResult(
                classification=result['classification'],
                confidence=result['confidence'],
                media_type=result['media_type']
            )
        else:
            raise HTTPException(status_code=400, detail=result.get('error', 'Detection failed'))
    
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/detect/audio", response_model=DetectionResult)
async def detect_audio(file: UploadFile = File(...)):
    if not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name
    
    try:
        result = audio_detector.detect(tmp_path)
        
        if result.get('status') == 'success':
            return DetectionResult(
                classification=result['classification'],
                confidence=result['confidence'],
                media_type=result['media_type']
            )
        else:
            raise HTTPException(status_code=400, detail=result.get('error', 'Detection failed'))
    
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/detect/video", response_model=DetectionResult)
async def detect_video(file: UploadFile = File(...), sample_rate: int = 10, 
                      visual_weight: float = 0.6):
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name
    
    try:
        result = video_detector.detect(tmp_path, sample_rate=sample_rate,
                                      fusion_weight_visual=visual_weight)
        
        if result.get('status') == 'success':
            return DetectionResult(
                classification=result['classification'],
                confidence=result['final_confidence'],
                media_type=result['media_type'],
                has_audio=result['has_audio'],
                visual_score=result['visual_score'],
                audio_score=result['audio_score'],
                temporal_consistency=result['temporal_consistency']
            )
        else:
            raise HTTPException(status_code=400, detail=result.get('error', 'Detection failed'))
    
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/models")
async def get_models_info():
    return {
        "image_detector": {
            "architecture": "ResNet50",
            "accuracy": "94.7%",
            "input_size": "224x224"
        },
        "audio_detector": {
            "architecture": "CNN-LSTM",
            "f1_score": "91.4%",
            "sample_rate": "16000 Hz"
        },
        "video_detector": {
            "visual_pipeline": "ResNet50 + temporal consistency",
            "audio_pipeline": "CNN-LSTM",
            "fusion": "60% visual, 40% audio"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
