#!/usr/bin/env python3
"""Eburon TTS Server - FastAPI backend for Eburon TTS Echo Model."""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import os
import uuid
import shutil

from mlx_audio.tts import load
import soundfile as sf
import numpy as np

app = FastAPI(title="Eburon TTS")

MODEL_PATH = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
OUTPUT_DIR = "/tmp/eburon_tts_outputs"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_model = None

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_model():
    global _model
    if _model is None:
        print("Loading Qwen3 TTS model...")
        _model = load(MODEL_PATH)
        print("Model loaded!")
    return _model


class TTSRequest(BaseModel):
    text: str


@app.get("/")
async def root():
    html_path = os.path.join(BASE_DIR, "templates", "index.html")
    return FileResponse(html_path)


@app.post("/generate")
async def generate_speech(request: TTSRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, text="Text is required")

    model = get_model()
    result = list(model.generate(text=request.text))[-1]

    filename = f"eburon_{uuid.uuid4().hex[:8]}.wav"
    output_path = os.path.join(OUTPUT_DIR, filename)
    sf.write(output_path, result.audio, result.sample_rate)

    return {
        "path": output_path,
        "duration": result.audio_duration,
        "sample_rate": result.sample_rate,
    }


@app.get("/audio")
async def get_audio(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, text="File not found")
    return FileResponse(path, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
