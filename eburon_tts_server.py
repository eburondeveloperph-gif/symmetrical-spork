#!/usr/bin/env python3
"""Eburon TTS Server - FastAPI backend for Eburon TTS Echo Model."""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional
import os
import uuid

from mlx_audio.tts import load
import soundfile as sf
import numpy as np

app = FastAPI(title="Eburon TTS")

MODEL_PATH = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
OUTPUT_DIR = "/tmp/eburon_tts_outputs"

_model = None

LANGUAGE_LEXICON = {
    "en": {"name": "English", "sample": "Hello! This is a test of Eburon TTS."},
    "nl": {
        "name": "Dutch (Flemish)",
        "sample": "Hallo! Dit is een test van Eburon TTS.",
    },
    "tl": {"name": "Tagalog", "sample": "Kamusta! Ito ay isang test ng Eburon TTS."},
}

VOICE_PRESETS = {
    "echo": {"speed": 1.0, "pitch": 0},
    "nova": {"speed": 1.1, "pitch": 2},
    "shell": {"speed": 0.95, "pitch": -1},
    "wave": {"speed": 1.05, "pitch": 1},
    "amber": {"speed": 0.9, "pitch": 0},
    "floyd": {"speed": 1.0, "pitch": 2},
}

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
    voice: Optional[str] = "echo"
    language: Optional[str] = "en"


@app.get("/")
async def root():
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    return FileResponse(html_path)


@app.get("/lexicon")
async def get_lexicon():
    """Get available languages and voices."""
    return {"languages": LANGUAGE_LEXICON, "voices": VOICE_PRESETS}


@app.post("/generate")
async def generate_speech(request: TTSRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, text="Text is required")

    voice = request.voice if request.voice in VOICE_PRESETS else "echo"
    language = request.language if request.language in LANGUAGE_LEXICON else "en"

    voice_params = VOICE_PRESETS[voice]

    model = get_model()

    generation_kwargs = {
        "text": request.text,
    }

    result = list(model.generate(**generation_kwargs))[-1]

    filename = f"eburon_{voice}_{uuid.uuid4().hex[:8]}.wav"
    output_path = os.path.join(OUTPUT_DIR, filename)
    sf.write(output_path, result.audio, result.sample_rate)

    return {
        "path": output_path,
        "duration": result.audio_duration,
        "sample_rate": result.sample_rate,
        "voice": voice,
        "language": language,
    }


@app.get("/audio")
async def get_audio(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, text="File not found")
    return FileResponse(path, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
