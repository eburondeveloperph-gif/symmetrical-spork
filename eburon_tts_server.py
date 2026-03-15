#!/usr/bin/env python3
"""Eburon TTS Server - FastAPI backend for Eburon TTS Echo Model.
Real-time TTS using Qwen3 model with natural human nuances (emotion, prosody, style).
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import uuid
import io
import re
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

# Language lexicon
LANGUAGE_LEXICON = {
    "en": {"name": "English", "sample": "Hello! This is a test of Eburon TTS."},
    "nl": {
        "name": "Dutch (Flemish)",
        "sample": "Hallo! Dit is een test van Eburon TTS.",
    },
    "tl": {"name": "Tagalog", "sample": "Kamusta! Ito ay isang test ng Eburon TTS."},
}

# Voice presets
VOICE_PRESETS = {
    "echo": {"speed": 1.0, "pitch": 0},
    "nova": {"speed": 1.1, "pitch": 2},
    "shell": {"speed": 0.95, "pitch": -1},
    "wave": {"speed": 1.05, "pitch": 1},
    "amber": {"speed": 0.9, "pitch": 0},
    "floyd": {"speed": 1.0, "pitch": 2},
}

# Emotion mappings for natural nuances
EMOTION_PROMPTS = {
    "happy": "Happy and cheerful tone, with upward inflections",
    "sad": "Sad and melancholic tone, slower pace, lower pitch",
    "angry": "Angry and frustrated tone, faster pace, harsh emphasis",
    "excited": "Excited and enthusiastic tone, fast pace, high energy",
    "calm": "Calm and soothing tone, relaxed pace, gentle delivery",
    "surprised": "Surprised and amazed tone, with natural exclamation",
    "scared": "Scared and nervous tone, hesitant, shaky delivery",
}

# Style modifiers for prosody control
STYLE_MODIFIERS = {
    "narrator": "Professional narrator voice, clear enunciation",
    "news": "News anchor delivery, formal and authoritative",
    "casual": "Casual conversation, relaxed and friendly",
    "dramatic": "Dramatic storytelling, theatrical pauses",
    "whisper": "Quiet whisper, intimate and soft",
    "shout": "Loud and emphatic, strong projection",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_model():
    global _model
    if _model is None:
        print("Loading Qwen3 TTS model for natural nuances...")
        _model = load(MODEL_PATH)
        print("Model loaded! Ready for natural TTS.")
    return _model


def enhance_text_with_nuances(text: str, emotion: str = "", style: str = "") -> str:
    """Enhance text with natural pauses and nuances for more human-like output."""

    # Add punctuation-based pauses for natural rhythm
    # These help the TTS model understand phrasing
    text = re.sub(r"([.!?])\1+", r"\1", text)  # Clean repeated punctuation

    # Add ellipsis pauses for dramatic effect if emotion warrants
    if emotion in ["dramatic", "sad", "scared"]:
        text = re.sub(r"(\.{3,})", r" \1 ", text)

    return text


def build_nuance_prompt(emotion: str = "", style: str = "") -> str:
    """Build a natural language prompt for nuanced TTS generation."""
    prompts = []

    if emotion and emotion in EMOTION_PROMPTS:
        prompts.append(EMOTION_PROMPTS[emotion])

    if style:
        style_lower = style.lower()
        for key, modifier in STYLE_MODIFIERS.items():
            if key in style_lower:
                prompts.append(modifier)
                break
        else:
            # Use custom style description
            prompts.append(style)

    return ", ".join(prompts) if prompts else ""


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "echo"
    language: Optional[str] = "en"
    emotion: Optional[str] = ""
    style: Optional[str] = ""


@app.get("/")
async def root():
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    return FileResponse(html_path)


@app.get("/lexicon")
async def get_lexicon():
    return {
        "languages": LANGUAGE_LEXICON,
        "voices": VOICE_PRESETS,
        "emotions": EMOTION_PROMPTS,
        "styles": STYLE_MODIFIERS,
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "Qwen3-TTS (MLX)",
        "realtime": True,
        "nuances": True,
    }


@app.post("/generate")
async def generate_speech(request: TTSRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, text="Text is required")

    voice = request.voice if request.voice in VOICE_PRESETS else "echo"
    language = request.language if request.language in LANGUAGE_LEXICON else "en"
    emotion = request.emotion if request.emotion in EMOTION_PROMPTS else ""
    style = request.style

    # Build nuanced prompt for natural speech
    nuance_prompt = build_nuance_prompt(emotion, style)

    # Enhance text with pauses for natural rhythm
    enhanced_text = enhance_text_with_nuances(request.text, emotion, style)

    # Combine text with nuance instructions if available
    final_text = enhanced_text
    if nuance_prompt:
        final_text = f"[{nuance_prompt}] {enhanced_text}"

    model = get_model()
    result = list(model.generate(text=final_text))[-1]

    filename = f"eburon_{voice}_{emotion}_{uuid.uuid4().hex[:8]}.wav"
    output_path = os.path.join(OUTPUT_DIR, filename)
    sf.write(output_path, result.audio, result.sample_rate)

    return {
        "path": output_path,
        "duration": result.audio_duration,
        "sample_rate": result.sample_rate,
        "voice": voice,
        "language": language,
        "emotion": emotion,
        "style": style,
    }


@app.post("/generate/stream")
async def generate_stream(request: TTSRequest):
    """Real-time streaming TTS with natural nuances."""
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, text="Text is required")

    nuance_prompt = build_nuance_prompt(request.emotion, request.style)
    enhanced_text = enhance_text_with_nuances(
        request.text, request.emotion, request.style
    )
    final_text = (
        f"[{nuance_prompt}] {enhanced_text}" if nuance_prompt else enhanced_text
    )

    model = get_model()

    async def generate():
        buffer = io.BytesIO()
        for result in model.generate(text=final_text):
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
