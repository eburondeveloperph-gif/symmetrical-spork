#!/usr/bin/env python3
"""Eburon TTS Server - FastAPI backend for Eburon TTS Echo Model.
Real-time TTS using Qwen3 0.6B model via MLX with streaming.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import uuid
import io
import asyncio
import numpy as np
from mlx_audio.tts import load
import soundfile as sf

app = FastAPI(title="Eburon TTS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        print("Loading Qwen3 TTS model (0.6B equivalent)...")
        _model = load(MODEL_PATH)
        print("Model loaded! Ready for real-time TTS.")
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
    return {"languages": LANGUAGE_LEXICON, "voices": VOICE_PRESETS}


@app.get("/health")
async def health():
    return {"status": "ok", "model": "Qwen3-TTS-0.6B (MLX)", "realtime": True}


@app.post("/generate")
async def generate_speech(request: TTSRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, text="Text is required")

    voice = request.voice if request.voice in VOICE_PRESETS else "echo"
    language = request.language if request.language in LANGUAGE_LEXICON else "en"

    model = get_model()
    result = list(model.generate(text=request.text))[-1]

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


@app.post("/generate/stream")
async def generate_stream(request: TTSRequest):
    """Real-time streaming TTS - audio starts playing immediately."""
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, text="Text is required")

    model = get_model()

    async def generate():
        buffer = io.BytesIO()

        for result in model.generate(text=request.text):
            if hasattr(result, "audio") and result.audio is not None:
                audio = result.audio
                sf.write(buffer, audio, result.sample_rate, format="WAV")
                buffer.seek(0)
                yield buffer.read()
                buffer.seek(0)
                buffer.truncate()

    return StreamingResponse(generate(), media_type="audio/wav")


@app.get("/audio")
async def get_audio(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, text="File not found")
    return FileResponse(path, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
