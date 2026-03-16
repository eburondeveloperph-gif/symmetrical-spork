#!/usr/bin/env python3
"""Eburon TTS Server - FastAPI backend for Eburon TTS Echo Model.
Real-time TTS using Qwen3 model with natural human nuances (emotion, prosody, style).
Voice cloning support added.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import os
import uuid
import io
import re
import json
import sqlite3
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

MODEL_PATH = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
OUTPUT_DIR = "/tmp/eburon_tts_outputs"
VOICE_PROMPTS_DIR = (
    "/Users/master/vbox/voicebox/scripts/training_data/itawit/voice_clones"
)
ITAWIT_MODELS_DIR = "/Users/master/vbox/voicebox/scripts/training_data/itawit/models"
DB_PATH = "/tmp/echobox.db"

_model = None
_voice_prompts: Dict[str, dict] = {}


def load_persistent_voice_prompts():
    """Load voice prompts from persistent storage on startup."""
    global _voice_prompts
    if os.path.exists(VOICE_PROMPTS_DIR):
        for f in os.listdir(VOICE_PROMPTS_DIR):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(VOICE_PROMPTS_DIR, f), "r") as fp:
                        prompt_data = json.load(fp)
                        prompt_id = prompt_data.get("id")
                        if prompt_id:
                            _voice_prompts[prompt_id] = prompt_data
                            print(
                                f"Loaded voice prompt: {prompt_data.get('name')} ({prompt_id})"
                            )
                except Exception as e:
                    print(f"Error loading voice prompt {f}: {e}")


# Load persistent voice prompts on module load
load_persistent_voice_prompts()


def init_db():
    """Initialize SQLite database with tables."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS generations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            voice TEXT,
            emotion TEXT,
            style TEXT,
            language TEXT,
            audio_path TEXT,
            duration REAL,
            voice_clone TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS voice_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            audio_path TEXT NOT NULL,
            reference_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            language TEXT NOT NULL,
            audio_path TEXT NOT NULL,
            transcribe_text TEXT,
            duration REAL,
            source TEXT,
            is_used INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS itawit_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            model_path TEXT,
            training_epochs INTEGER DEFAULT 0,
            num_samples INTEGER DEFAULT 0,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_generation(
    text: str,
    voice: str,
    emotion: str,
    style: str,
    language: str,
    audio_path: str,
    duration: float,
    voice_clone: str = None,
):
    """Save a TTS generation to database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO generations (text, voice, emotion, style, language, audio_path, duration, voice_clone)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (text, voice, emotion, style, language, audio_path, duration, voice_clone),
        )
        conn.commit()
        gen_id = c.lastrowid
        conn.close()
        return gen_id
    except Exception as e:
        print(f"Error saving generation: {e}")
        return None


def get_generations(limit: int = 50):
    """Get recent generations from database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            """
            SELECT id, text, voice, emotion, style, language, audio_path, duration, voice_clone, created_at
            FROM generations ORDER BY created_at DESC LIMIT ?
        """,
            (limit,),
        )
        rows = c.fetchall()
        conn.close()
        return [
            {
                "id": r[0],
                "text": r[1],
                "voice": r[2],
                "emotion": r[3],
                "style": r[4],
                "language": r[5],
                "audio_path": r[6],
                "duration": r[7],
                "voice_clone": r[8],
                "created_at": r[9],
            }
            for r in rows
        ]
    except Exception as e:
        print(f"Error getting generations: {e}")
        return []


def save_voice_profile(name: str, audio_path: str, reference_text: str = ""):
    """Save a voice profile to database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO voice_profiles (name, audio_path, reference_text)
            VALUES (?, ?, ?)
        """,
            (name, audio_path, reference_text),
        )
        conn.commit()
        profile_id = c.lastrowid
        conn.close()
        return profile_id
    except Exception as e:
        print(f"Error saving voice profile: {e}")
        return None


def get_voice_profiles():
    """Get all voice profiles from database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT id, name, audio_path, reference_text, created_at
            FROM voice_profiles ORDER BY created_at DESC
        """)
        rows = c.fetchall()
        conn.close()
        return [
            {
                "id": r[0],
                "name": r[1],
                "audio_path": r[2],
                "reference_text": r[3],
                "created_at": r[4],
            }
            for r in rows
        ]
    except Exception as e:
        print(f"Error getting voice profiles: {e}")
        return []


def delete_voice_profile(profile_id: int):
    """Delete a voice profile from database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM voice_profiles WHERE id = ?", (profile_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error deleting voice profile: {e}")
        return False


# Initialize database on startup
init_db()

# Language lexicon - all Qwen3 supported + custom
LANGUAGE_LEXICON = {
    # Qwen3 native support
    "auto": {"name": "Auto Detect", "sample": ""},
    "en": {"name": "English", "sample": "Hello! This is a test of Eburon TTS."},
    "zh": {"name": "Chinese", "sample": "你好！这是Eburon TTS的测试。"},
    "de": {"name": "German", "sample": "Hallo! Das ist ein Test von Eburon TTS."},
    "it": {"name": "Italian", "sample": "Ciao! Questo è un test di Eburon TTS."},
    "pt": {"name": "Portuguese", "sample": "Olá! Este é um teste do Eburon TTS."},
    "es": {"name": "Spanish", "sample": "¡Hola! Esta es una prueba de Eburon TTS."},
    "ja": {"name": "Japanese", "sample": "こんにちは！これはEburon TTSのテストです。"},
    "ko": {"name": "Korean", "sample": "안녕하세요! 이것은 Eburon TTS 테스트입니다."},
    "fr": {"name": "French", "sample": "Bonjour! C'est un test d'Eburon TTS."},
    "ru": {"name": "Russian", "sample": "Привет! Это тест Eburon TTS."},
    # Custom - requires training
    "nl": {
        "name": "Dutch (Flemish) - Training",
        "sample": "Hallo! Dit is een test van Eburon TTS.",
        "needs_training": True,
    },
    "tl": {
        "name": "Tagalog - Training",
        "sample": "Kamusta! Ito ay isang test ng Eburon TTS.",
        "needs_training": True,
    },
    "itw": {
        "name": "Itawit - Training",
        "sample": "Ma-ngo! Mabbalat.",
        "needs_training": True,
        "training_data": "/Users/master/vbox/voicebox/scripts/training_data/itawit",
        # Correct lexicon based on JW.org Itawit publications and native speakers
        "lexicon": {
            # Greetings & Common Phrases (verified Itawit)
            "ma_ngo": "Hello",
            "mabbalat": "Thank you",
            "oon": "Yes",
            "awan": "No",
            "napia": "Good/Well",
            "napia_nga_algaw": "Good day",
            "napia_nga_mataruk": "Good morning",
            "napia_nga_giram": "Good afternoon",
            "napia_nga_gabi": "Good night",
            "kunnasi_ka": "How are you?",
            "minya_ka": "How are you?",
            "inaria_ko": "I don't know",
            "ammuk": "I know",
            "marik_ammuk": "I don't know",
            "baka": "Maybe",
            "siguru": "Definitely",
            "pay_e": "Please",
            "innam_mu_ikau": "Take care",
            "ay_ayatan_ta_ka": "I love you",
            "anna_yo_ngahan_mu": "What is your name?",
            "yo_ngahan_ku_e": "My name is...",
            "napia_nga_nakilala": "Nice to meet you",
            # Bible Terms from JW.org Itawit
            "biblia": "Bible",
            "jehova": "Jehovah",
            "dios": "God",
            "kristiano": "Christian",
            "saksi": "Witness",
            "kongregasion": "Congregation",
            "mangangampet": "Prayer",
            "sarita": "Word/Story",
            "baluan": "Verse",
            "kaagian": "Church/Meeting place",
            # JW.org Publication Terms
            "balita": "News/Message",
            "attolay": "People",
            "mannanayun": "Mankind",
            "mappagilammu": "Knowledge",
            "paggilammu": "Understanding",
            "nakakkasta": "Good News",
            # Time
            "algaw": "Day",
            "bigat": "Morning",
            "giram": "Afternoon",
            "gabi": "Night",
            "tugpet": "Noon",
            "sabado": "Saturday",
            "domingo": "Sunday",
            "bulan": "Month",
            "tawen": "Year",
            # Family
            "famillia": "Family",
            "ama": "Father",
            "ina": "Mother",
            "anak": "Child",
            "kabsat": "Sibling",
            "lalaki": "Male/Man",
            "babai": "Female/Woman",
            # Numbers
            "isa": "One",
            "duwa": "Two",
            "tallo": "Three",
            "uppat": "Four",
            "lima": "Five",
            "unnam": "Six",
            "pito": "Seven",
            "walo": "Eight",
            "siam": "Nine",
            "sapol": "Ten",
        },
        # Sample texts from JW.org Itawit publications
        "sample_texts": [
            "Ma-ngo! Mabbalat.",
            "Kunnasi ka? Napia nak.",
            "Anna yo ngahan mu? Yo ngahan ku e Maria.",
            "Ay-ayatan ta ka.",
            "Ti biblian ti kasisirinan na.",
            "Ti mappagilammu ket importanti para ittayo.",
            "Nakakkasta nga Balita nga Naggafu kan Dios!",
            "Napagayayyat nga Attolay kan Mannanayun!",
            "Maggina Ka kan Dios Tase nu Matolay Ka kan Mannanayun.",
            "Jehova i Dios.",
        ],
    },
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

# Emotion mappings for natural nuances - HIGH INTENSITY
EMOTION_PROMPTS = {
    "happy": "Joyful exuberant tone, bright cheerful voice, infectious happiness, warm radiant delivery, animated and lively",
    "sad": "Deep sorrowful tone, heavy heartbroken voice, devastating grief, mournful melancholic delivery, profound sadness",
    "angry": "Furious raging tone, fierce aggressive voice, intense burning anger, seething with rage, forceful powerful delivery",
    "excited": "Electrifying enthusiastic tone,激动不已的声音, exploding with excitement, breathless anticipation, high-energy electrified delivery",
    "calm": "Serene peaceful tone, gentle tranquil voice, deep inner peace, soothing mellow delivery, tranquil relaxed atmosphere",
    "fear": "Terrified trembling tone, scared frightened voice, panic-stricken, nervous shaky delivery, anxious terrified whisper",
    "disgust": "Repulsed revolted tone, disgusted contemptuous voice, extreme aversion, nauseated delivery, sickened repulsed tone",
    "surprised": "Stunned shocked tone, completely bewildered voice, jaw-dropping astonishment, utterly amazed delivery, blown-away reaction",
    "neutral": "Natural conversational tone, regular speaking voice, balanced and steady delivery, clear and neutral",
}

# Style modifiers for prosody control
STYLE_MODIFIERS = {
    "narrator": "Professional narrator voice, clear enunciation, storytelling mode",
    "news": "News anchor delivery, formal authoritative, measured cadence",
    "casual": "Casual conversation, relaxed informal, friendly chat",
    "dramatic": "Dramatic theatrical delivery, exaggerated pauses, storytelling intensity",
    "whisper": "Secretive whisper, hushed intimate voice, quiet conspiratorial",
    "shout": "LOUD shouting, powerful projection, commanding roar",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VOICE_PROMPTS_DIR, exist_ok=True)


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
    voice_prompt_id: Optional[str] = None  # For voice cloning
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


@app.get("/lexicon/{lang_code}")
async def get_language_lexicon(lang_code: str):
    """Get detailed lexicon for a specific language including sample texts"""
    if lang_code not in LANGUAGE_LEXICON:
        raise HTTPException(status_code=404, detail="Language not found")

    lang = LANGUAGE_LEXICON[lang_code]
    return {
        "code": lang_code,
        "name": lang.get("name"),
        "sample": lang.get("sample"),
        "needs_training": lang.get("needs_training", False),
        "training_data": lang.get("training_data"),
        "lexicon": lang.get("lexicon", {}),
        "sample_texts": lang.get("sample_texts", []),
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

    # Check for voice cloning
    voice_prompt = None
    if request.voice_prompt_id and request.voice_prompt_id in _voice_prompts:
        voice_prompt = _voice_prompts[request.voice_prompt_id]
        print(f"Using voice clone: {voice_prompt['name']}")

    # Build nuanced prompt for natural speech
    nuance_prompt = build_nuance_prompt(emotion or "", style or "")

    # Enhance text with pauses for natural rhythm
    enhanced_text = enhance_text_with_nuances(request.text, emotion or "", style or "")

    # Combine text with nuance instructions if available
    final_text = enhanced_text
    if nuance_prompt:
        final_text = f"[{nuance_prompt}] {enhanced_text}"

    model = get_model()

    # Use generate_voice_design with instruct for emotion/style control
    generation_kwargs = {"text": request.text}

    # Handle Itawit language with training data
    itawit_training_dir = "/Users/master/vbox/voicebox/scripts/training_data/itawit"
    # Itawit-specific voice description for native-like output
    itawit_voice_desc = (
        "Itawit language speaker from Cagayan Valley, Philippines. "
        "Northern Luzon Philippine accent. Clear enunciation, natural Philippine speech patterns. "
        "Medium pace, slightly melodic intonation typical of Itawit speakers."
    )

    if language == "itw" and os.path.exists(itawit_training_dir):
        # Find reference audio in training data
        training_audio = None
        for f in os.listdir(itawit_training_dir):
            if f.endswith((".wav", ".mp3", ".flac")):
                training_audio = os.path.join(itawit_training_dir, f)
                break

        if training_audio and os.path.exists(training_audio):
            try:
                # For Itawit, use the training audio name in the instruct to help with voice cloning
                training_name = os.path.basename(training_audio)
                itawit_instruct = (
                    f"Speak in Itawit language ({itawit_voice_desc}). "
                    f"Reference voice from: {training_name}. "
                )
                # Prepend Itawit instruction to the instruct prompt
                if "instruct" in generation_kwargs:
                    generation_kwargs["instruct"] = (
                        itawit_instruct + generation_kwargs["instruct"]
                    )

                print(f"Applied Itawit training data from: {training_audio}")
            except Exception as e:
                print(f"Warning: Failed to apply Itawit training: {e}")
        else:
            # Still add Itawit voice description even without training audio
            if "instruct" in generation_kwargs:
                generation_kwargs["instruct"] = (
                    f"Speak in Itawit language ({itawit_voice_desc}). "
                    + generation_kwargs["instruct"]
                )

    if voice_prompt:
        # Use voice clone prompt
        ref_audio_path = voice_prompt.get("audio_path")
        ref_text = voice_prompt.get("reference_text", "")

        if ref_audio_path and os.path.exists(ref_audio_path):
            try:
                # Load reference audio
                ref_audio, ref_sr = sf.read(ref_audio_path)

                # Use create_voice_clone_prompt if available
                if hasattr(model, "create_voice_clone_prompt"):
                    voice_clone_prompt = model.create_voice_clone_prompt(
                        ref_audio, ref_text
                    )
                    generation_kwargs["voice_clone_prompt"] = voice_clone_prompt
                else:
                    # Fallback: use ref_audio as reference - build into instruct instead
                    # Note: Qwen3-TTS uses instruct for voice design, not ref_audio
                    voice_clone_instruct = (
                        f"Clone voice from reference: {ref_text}. "
                        f"Match the voice characteristics from the reference audio."
                    )
                    # Prepend voice clone instruction to the instruct prompt
                    if "instruct" in generation_kwargs:
                        generation_kwargs["instruct"] = (
                            voice_clone_instruct + " " + generation_kwargs["instruct"]
                        )

                print(f"Applied voice clone from: {ref_audio_path}")
            except Exception as e:
                print(f"Warning: Failed to apply voice clone: {e}")

    # Always use voice design with instruct parameter (required by Qwen3)
    voice_descriptions = {
        "echo": "A neutral adult male voice with medium pitch, clear and professional",
        "nova": "A energetic young female voice with higher pitch, warm and friendly",
        "shell": "A calm adult female voice with lower pitch, soft and soothing",
        "wave": "A balanced adult male voice with medium pitch, natural and expressive",
        "amber": "A gentle adult female voice with medium-low pitch, warm and caring",
        "floyd": "A confident adult male voice with higher pitch, bold and dynamic",
    }

    # Build instruct from emotion/style or use default voice description
    # Add emotion-specific speed and pitch for more intense output
    emotion_speed = {"pitch": 1.0}

    if emotion or style:
        instruct = build_nuance_prompt(emotion or "", style or "")

        # Add emotion-specific prosody adjustments for intensity
        if emotion == "angry":
            emotion_speed = {"speed": 1.15, "pitch": 1.3}
            instruct = (
                "ANGRY FURIOUS RAGING VOICE: "
                + instruct
                + " speak with INTENSE RAW UNCONTROLLED EMOTION"
            )
        elif emotion == "happy":
            emotion_speed = {"speed": 1.2, "pitch": 1.2}
            instruct = (
                "HAPPY JOYFUL EXCITED VOICE: "
                + instruct
                + " speak with RADIANT EUPHORIC ENERGY"
            )
        elif emotion == "sad":
            emotion_speed = {"speed": 0.75, "pitch": 0.8}
            instruct = (
                "SAD MELANCHOLIC HEAVY HEART VOICE: "
                + instruct
                + " speak with DEVASTATING GRIEF AND SORROW"
            )
        elif emotion == "excited":
            emotion_speed = {"speed": 1.3, "pitch": 1.4}
            instruct = (
                "EXCITED ELECTRIFIED MANIC VOICE: "
                + instruct
                + " speak with FRANTIC ENERGY AND HYSTERICAL EXCITEMENT"
            )
        elif emotion == "calm":
            emotion_speed = {"speed": 0.85, "pitch": 0.9}
            instruct = (
                "CALM SERENE TRANQUIL VOICE: "
                + instruct
                + " speak with PEACEFUL GENTLE SOFTNESS"
            )
        elif emotion == "fear":
            emotion_speed = {"speed": 1.4, "pitch": 1.5}
            instruct = (
                "TERRIFIED PANIC-STRICKEN SCREAMING VOICE: "
                + instruct
                + " speak with PARANOID FEAR AND TERROR"
            )
        elif emotion == "disgust":
            emotion_speed = {"speed": 0.7, "pitch": 1.1}
            instruct = (
                "DISGUSTED REVOLTED SICKENED VOICE: "
                + instruct
                + " speak with EXTREME REPULSION AND NAUSEA"
            )
        elif emotion == "surprised":
            emotion_speed = {"speed": 1.25, "pitch": 1.35}
            instruct = (
                "SHOCKED STUNNED BLOWN AWAY VOICE: "
                + instruct
                + " speak with JAW-DROPPING ASTONISHMENT"
            )
    else:
        instruct = voice_descriptions.get(voice, voice_descriptions["echo"])

    generation_kwargs["instruct"] = instruct

    # Apply emotion prosody if specified
    if emotion and emotion in emotion_speed:
        if "speed" in emotion_speed:
            generation_kwargs["speed"] = emotion_speed["speed"]
        if "pitch" in emotion_speed:
            generation_kwargs["pitch"] = emotion_speed["pitch"]

    result = list(model.generate_voice_design(**generation_kwargs))[-1]

    filename = f"eburon_{voice}_{emotion}_{uuid.uuid4().hex[:8]}.wav"
    output_path = os.path.join(OUTPUT_DIR, filename)
    sf.write(output_path, result.audio, result.sample_rate)

    # Save to database
    gen_id = save_generation(
        text=request.text,
        voice=voice,
        emotion=emotion,
        style=style,
        language=language,
        audio_path=output_path,
        duration=result.audio_duration,
        voice_clone=voice_prompt["name"] if voice_prompt else None,
    )

    return {
        "path": output_path,
        "duration": result.audio_duration,
        "sample_rate": result.sample_rate,
        "voice": voice,
        "voice_clone": voice_prompt["name"] if voice_prompt else None,
        "language": language,
        "emotion": emotion,
        "style": style,
        "db_id": gen_id,
    }


@app.post("/generate/stream")
async def generate_stream(request: TTSRequest):
    """Real-time streaming TTS with natural nuances."""
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, text="Text is required")

    model = get_model()

    generation_kwargs = {"text": request.text}

    if request.emotion or request.style:
        instruct = build_nuance_prompt(request.emotion, request.style)
        generation_kwargs["instruct"] = instruct

    async def generate():
        buffer = io.BytesIO()
        if request.emotion or request.style:
            for result in model.generate_voice_design(**generation_kwargs):
                if hasattr(result, "audio") and result.audio is not None:
                    audio = result.audio
                    sf.write(buffer, audio, result.sample_rate, format="WAV")
                    buffer.seek(0)
                    yield buffer.read()
                    buffer.seek(0)
                    buffer.truncate()
        else:
            for result in model.generate(**generation_kwargs):
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


# Voice Cloning Endpoints


@app.post("/voice/clone")
async def clone_voice(
    audio: UploadFile = File(...),
    reference_text: str = Form(...),
    name: str = Form("Cloned Voice"),
):
    """Upload audio sample to create a voice clone."""
    try:
        # Read audio file
        audio_data = await audio.read()

        # Save to temp file
        temp_path = os.path.join(
            VOICE_PROMPTS_DIR, f"{uuid.uuid4().hex}_{audio.filename}"
        )
        with open(temp_path, "wb") as f:
            f.write(audio_data)

        # Load and validate audio
        try:
            audio_array, sr = sf.read(io.BytesIO(audio_data))

            # Resample to 24kHz if needed
            if sr != 24000:
                try:
                    import librosa

                    audio_array = librosa.resample(
                        audio_array, orig_sr=sr, target_sr=24000
                    )
                    sr = 24000
                except ImportError:
                    pass  # Skip resampling if librosa not available

        except Exception as e:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise HTTPException(status_code=400, text=f"Invalid audio file: {str(e)}")

        # Create voice prompt
        voice_prompt_id = f"voice_{uuid.uuid4().hex[:12]}"

        voice_prompt = {
            "id": voice_prompt_id,
            "name": name,
            "audio_path": temp_path,
            "reference_text": reference_text,
            "sample_rate": sr,
        }

        _voice_prompts[voice_prompt_id] = voice_prompt

        # Save to persistent storage
        try:
            os.makedirs(VOICE_PROMPTS_DIR, exist_ok=True)
            prompt_file = os.path.join(VOICE_PROMPTS_DIR, f"{voice_prompt_id}.json")
            with open(prompt_file, "w") as fp:
                json.dump(voice_prompt, fp)
            print(f"Saved voice prompt to: {prompt_file}")
        except Exception as e:
            print(f"Warning: Failed to save voice prompt to persistent storage: {e}")

        # Save to database
        profile_id = save_voice_profile(name, temp_path, reference_text)

        return {
            "success": True,
            "voice_prompt_id": voice_prompt_id,
            "name": name,
            "reference_text": reference_text,
            "duration": len(audio_array) / sr if sr else 0,
            "db_id": profile_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, text=f"Failed to clone voice: {str(e)}")


@app.get("/voice/prompts")
async def list_voice_prompts():
    """List all voice prompts."""
    return {
        "voice_prompts": [
            {
                "id": vp["id"],
                "name": vp["name"],
                "reference_text": vp["reference_text"][:100] + "..."
                if len(vp.get("reference_text", "")) > 100
                else vp.get("reference_text", ""),
            }
            for vp in _voice_prompts.values()
        ]
    }


@app.delete("/voice/prompts/{prompt_id}")
async def delete_voice_prompt(prompt_id: str):
    """Delete a voice prompt."""
    if prompt_id not in _voice_prompts:
        raise HTTPException(status_code=404, text="Voice prompt not found")

    voice_prompt = _voice_prompts[prompt_id]

    # Remove audio file
    if os.path.exists(voice_prompt.get("audio_path", "")):
        try:
            os.remove(voice_prompt["audio_path"])
        except:
            pass

    # Remove from persistent storage
    prompt_file = os.path.join(VOICE_PROMPTS_DIR, f"{prompt_id}.json")
    if os.path.exists(prompt_file):
        try:
            os.remove(prompt_file)
        except:
            pass

    del _voice_prompts[prompt_id]

    return {"success": True, "message": "Voice prompt deleted"}


@app.get("/voice/prompts/{prompt_id}/audio")
async def get_voice_prompt_audio(prompt_id: str):
    """Get the audio for a voice prompt."""
    if prompt_id not in _voice_prompts:
        raise HTTPException(status_code=404, text="Voice prompt not found")

    voice_prompt = _voice_prompts[prompt_id]
    audio_path = voice_prompt.get("audio_path", "")

    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, text="Audio file not found")

    return FileResponse(audio_path, media_type="audio/wav")


# Database API Endpoints
@app.get("/db/generations")
async def get_db_generations(limit: int = 50):
    """Get TTS generation history from database."""
    return {"generations": get_generations(limit)}


@app.get("/db/generations/{gen_id}")
async def get_db_generation(gen_id: int):
    """Get a specific generation by ID."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        SELECT id, text, voice, emotion, style, language, audio_path, duration, voice_clone, created_at
        FROM generations WHERE id = ?
    """,
        (gen_id,),
    )
    row = c.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, text="Generation not found")
    return {
        "id": row[0],
        "text": row[1],
        "voice": row[2],
        "emotion": row[3],
        "style": row[4],
        "language": row[5],
        "audio_path": row[6],
        "duration": row[7],
        "voice_clone": row[8],
        "created_at": row[9],
    }


@app.delete("/db/generations/{gen_id}")
async def delete_db_generation(gen_id: int):
    """Delete a generation from database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM generations WHERE id = ?", (gen_id,))
        conn.commit()
        conn.close()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, text=str(e))


@app.get("/db/voice-profiles")
async def get_db_voice_profiles():
    """Get all voice profiles from database."""
    return {"profiles": get_voice_profiles()}


@app.post("/db/voice-profiles")
async def create_db_voice_profile(request: dict):
    """Create a voice profile in database."""
    name = request.get("name")
    audio_path = request.get("audio_path")
    reference_text = request.get("reference_text", "")
    if not name or not audio_path:
        raise HTTPException(status_code=400, text="name and audio_path are required")
    profile_id = save_voice_profile(name, audio_path, reference_text)
    return {
        "id": profile_id,
        "name": name,
        "audio_path": audio_path,
        "reference_text": reference_text,
    }


@app.delete("/db/voice-profiles/{profile_id}")
async def delete_db_voice_profile(profile_id: int):
    """Delete a voice profile from database."""
    success = delete_voice_profile(profile_id)
    if not success:
        raise HTTPException(status_code=500, text="Failed to delete voice profile")
    return {"success": True}


@app.get("/db/stats")
async def get_db_stats():
    """Get database statistics."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM generations")
        gen_count = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM voice_profiles")
        profile_count = c.fetchone()[0]
        conn.close()
        return {"total_generations": gen_count, "total_voice_profiles": profile_count}
    except Exception as e:
        return {"error": str(e)}


# STT (Speech-to-Text) - Aquilles Model
STT_MODEL = None


def get_whisper_model():
    """Get or load Whisper STT model (Aquilles)."""
    global STT_MODEL
    if STT_MODEL is None:
        try:
            from mlx_audio.stt import load as load_stt

            print("Loading Whisper STT model (Aquilles)...")
            STT_MODEL = load_stt("mlx-community/whisper-base-mlx")
            print("Whisper STT model loaded!")
        except Exception as e:
            print(f"Failed to load MLX Whisper: {e}")
            try:
                import whisper

                STT_MODEL = whisper.load_model("base")
                print("Loaded PyTorch Whisper model as fallback")
            except Exception as e2:
                print(f"Failed to load Whisper: {e2}")
                raise HTTPException(
                    status_code=500, detail="Failed to load Whisper model"
                )
    return STT_MODEL


@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str = Form(None),
):
    """Transcribe audio to text using Eburon AI STT - Aquilles Model."""
    import tempfile
    import subprocess
    import os
    from pathlib import Path

    # Save uploaded file with proper extension
    original_ext = Path(audio.filename).suffix if audio.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=original_ext, delete=False) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Convert to 16kHz mono WAV using ffmpeg
    converted_path = tmp_path + "_converted.wav"
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                tmp_path,
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                converted_path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            # Try using original file
            converted_path = tmp_path

        # Get audio duration from converted file
        if os.path.exists(converted_path):
            audio_data, sr = sf.read(converted_path)
            duration = len(audio_data) / sr
            audio_path_for_model = converted_path
        else:
            audio_data, sr = sf.read(tmp_path)
            duration = len(audio_data) / sr
            audio_path_for_model = tmp_path

        # Transcribe
        whisper_model = get_whisper_model()

        try:
            decode_options = {}
            if language:
                decode_options["language"] = language

            print(f"Transcribing with path: {audio_path_for_model}")
            result = whisper_model.generate(str(audio_path_for_model), **decode_options)

            # Extract text from result
            if isinstance(result, str):
                text = result.strip()
            elif isinstance(result, dict):
                text = result.get("text", "").strip()
            elif hasattr(result, "text"):
                text = result.text.strip()
            else:
                text = str(result).strip()

        except Exception as e:
            print(f"Transcription error: {e}")
            import traceback

            traceback.print_exc()
            text = ""

        return {
            "text": text if text else "(No speech detected or transcription failed)",
            "duration": duration,
            "language": language or "auto-detected",
            "model": "Eburon AI STT - Aquilles Model",
        }

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        Path(converted_path).unlink(missing_ok=True)


@app.get("/stt/status")
async def get_stt_status():
    """Check STT model status."""
    global STT_MODEL
    return {
        "model_loaded": STT_MODEL is not None,
        "model_name": "Eburon AI STT - Aquilles Model (Whisper)",
    }


# Database API Endpoints


@app.post("/voice/save")
async def save_voice_clone(
    name: str = Form(...),
    audio_path: str = Form(...),
    reference_text: str = Form(""),
    language: str = Form("en"),
):
    """Save a voice clone to database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO voice_clones (name, audio_path, reference_text, language, created_at)
            VALUES (?, ?, ?, ?, datetime('now'))
        """,
            (name, audio_path, reference_text, language),
        )
        clone_id = c.lastrowid
        conn.commit()
        conn.close()
        return {"success": True, "clone_id": clone_id}
    except Exception as e:
        raise HTTPException(
            status_code=500, text=f"Failed to save voice clone: {str(e)}"
        )


@app.get("/voice/clones")
async def list_voice_clones():
    """List all saved voice clones."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT id, name, audio_path, reference_text, language, created_at
            FROM voice_clones ORDER BY created_at DESC
        """)
        clones = []
        for row in c.fetchall():
            clones.append(
                {
                    "id": row[0],
                    "name": row[1],
                    "audio_path": row[2],
                    "reference_text": row[3],
                    "language": row[4],
                    "created_at": row[5],
                }
            )
        conn.close()
        return {"voice_clones": clones}
    except Exception as e:
        raise HTTPException(
            status_code=500, text=f"Failed to list voice clones: {str(e)}"
        )


@app.delete("/voice/clones/{clone_id}")
async def delete_voice_clone(clone_id: int):
    """Delete a voice clone."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM voice_clones WHERE id = ?", (clone_id,))
        conn.commit()
        conn.close()
        return {"success": True}
    except Exception as e:
        raise HTTPException(
            status_code=500, text=f"Failed to delete voice clone: {str(e)}"
        )


@app.post("/training/add")
async def add_training_data(
    name: str = Form(...),
    language: str = Form(...),
    audio_path: str = Form(...),
    source: str = Form(""),
):
    """Add training data to database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO training_data (name, language, audio_path, source, created_at)
            VALUES (?, ?, ?, ?, datetime('now'))
        """,
            (name, language, audio_path, source),
        )
        training_id = c.lastrowid
        conn.commit()
        conn.close()
        return {"success": True, "training_id": training_id}
    except Exception as e:
        raise HTTPException(
            status_code=500, text=f"Failed to add training data: {str(e)}"
        )


@app.get("/training/list")
async def list_training_data(language: str = None):
    """List training data."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        if language:
            c.execute(
                """
                SELECT id, name, language, audio_path, source, created_at
                FROM training_data WHERE language = ? ORDER BY created_at DESC
            """,
                (language,),
            )
        else:
            c.execute("""
                SELECT id, name, language, audio_path, source, created_at
                FROM training_data ORDER BY created_at DESC
            """)
        training_data = []
        for row in c.fetchall():
            training_data.append(
                {
                    "id": row[0],
                    "name": row[1],
                    "language": row[2],
                    "audio_path": row[3],
                    "source": row[4],
                    "created_at": row[5],
                }
            )
        conn.close()
        return {"training_data": training_data}
    except Exception as e:
        raise HTTPException(
            status_code=500, text=f"Failed to list training data: {str(e)}"
        )


@app.post("/stt/save")
async def save_transcription(
    audio_path: str = Form(""),
    text: str = Form(...),
    language: str = Form(""),
    duration: float = Form(0),
):
    """Save a transcription to database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO transcriptions (audio_path, text, language, model, duration, created_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
        """,
            (audio_path, text, language, "Eburon AI STT - Aquilles Model", duration),
        )
        transcription_id = c.lastrowid
        conn.commit()
        conn.close()
        return {"success": True, "transcription_id": transcription_id}
    except Exception as e:
        raise HTTPException(
            status_code=500, text=f"Failed to save transcription: {str(e)}"
        )


@app.get("/stt/history")
async def list_transcriptions(limit: int = 50):
    """List transcription history."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            """
            SELECT id, audio_path, text, language, model, duration, created_at
            FROM transcriptions ORDER BY created_at DESC LIMIT ?
        """,
            (limit,),
        )
        transcriptions = []
        for row in c.fetchall():
            transcriptions.append(
                {
                    "id": row[0],
                    "audio_path": row[1],
                    "text": row[2],
                    "language": row[3],
                    "model": row[4],
                    "duration": row[5],
                    "created_at": row[6],
                }
            )
        conn.close()
        return {"transcriptions": transcriptions}
    except Exception as e:
        raise HTTPException(
            status_code=500, text=f"Failed to list transcriptions: {str(e)}"
        )


@app.get("/generations/history")
async def list_generations(limit: int = 50):
    """List TTS generation history."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            """
            SELECT id, text, voice, emotion, style, language, duration, created_at
            FROM generations ORDER BY created_at DESC LIMIT ?
        """,
            (limit,),
        )
        generations = []
        for row in c.fetchall():
            generations.append(
                {
                    "id": row[0],
                    "text": row[1],
                    "voice": row[2],
                    "emotion": row[3],
                    "style": row[4],
                    "language": row[5],
                    "duration": row[6],
                    "created_at": row[7],
                }
            )
        conn.close()
        return {"generations": generations}
    except Exception as e:
        raise HTTPException(
            status_code=500, text=f"Failed to list generations: {str(e)}"
        )


# Coqui XTTS Voice Cloning for Itawit
_coqui_tts = None
_coqui_model = None


def get_coqui_model():
    """Lazy load Coqui XTTS model"""
    global _coqui_tts, _coqui_model
    if _coqui_tts is None:
        try:
            from TTS.api import TTS

            print("Loading Coqui XTTS v2 model for Itawit voice cloning...")
            _coqui_tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            print("Coqui XTTS model loaded!")
        except Exception as e:
            print(f"Warning: Failed to load Coqui XTTS: {e}")
            _coqui_tts = False
    return _coqui_tts


@app.post("/generate/itawit")
async def generate_itawit_xtts(
    text: str = Form(...),
    reference_audio: Optional[str] = Form(None),
    use_multi_ref: bool = Form(False),
):
    """Generate Itawit speech using Coqui XTTS voice cloning with multi-reference support"""
    if not text or not text.strip():
        raise HTTPException(status_code=400, text="Text is required")

    model = get_coqui_model()
    if not model:
        raise HTTPException(
            status_code=500,
            text="Coqui XTTS model not available. Please install: pip install coqui-tts",
        )

    # Find reference audio
    itawit_dir = "/Users/master/vbox/voicebox/scripts/training_data/itawit"
    xtts_train_dir = os.path.join(itawit_dir, "xtts_train", "wav")
    segments_dir = os.path.join(itawit_dir, "segments")

    ref_audio_list = []

    # Use multiple reference audio files for better cloning
    if use_multi_ref:
        # First, check for processed segments (309+ Itawit segments from JW toolkit)
        if os.path.exists(segments_dir):
            segment_files = sorted(
                [f for f in os.listdir(segments_dir) if f.endswith(".wav")]
            )
            if segment_files:
                for f in segment_files[:20]:  # Use up to 20 segments for better cloning
                    ref_audio_list.append(os.path.join(segments_dir, f))
                print(f"Using {len(ref_audio_list)} Itawit segment references")

        # Fallback to xtts_train wav files
        if not ref_audio_list and os.path.exists(xtts_train_dir):
            for f in sorted(os.listdir(xtts_train_dir)):
                if f.endswith(".wav"):
                    ref_audio_list.append(os.path.join(xtts_train_dir, f))
            print(f"Using {len(ref_audio_list)} xtts_train reference audio files")

    # Single reference mode
    ref_audio = reference_audio
    if not ref_audio_list and (not ref_audio or not os.path.exists(ref_audio)):
        # Try to find default reference audio
        for f in ["kt_ITW_01_ref.wav", "kt_ITW_01 (1).wav", "itawit_reference.mp3"]:
            candidate = os.path.join(itawit_dir, f)
            if os.path.exists(candidate):
                ref_audio = candidate
                break

    if ref_audio_list:
        # Multi-reference mode
        pass  # Already populated
    elif ref_audio and os.path.exists(ref_audio):
        ref_audio_list = [ref_audio]
    else:
        raise HTTPException(
            status_code=400,
            text="Reference audio not found. Please provide a valid reference audio file.",
        )

    try:
        output_path = f"/tmp/itawit_xtts_{uuid.uuid4().hex[:8]}.wav"

        # Convert any MP3 to WAV
        converted_refs = []
        for ref in ref_audio_list:
            if ref.endswith(".mp3"):
                import subprocess

                wav_ref = output_path.replace(".wav", f"_{os.path.basename(ref)}.wav")
                subprocess.run(
                    ["ffmpeg", "-i", ref, "-ar", "24000", "-ac", "1", wav_ref, "-y"],
                    check=True,
                    capture_output=True,
                )
                converted_refs.append(wav_ref)
            else:
                converted_refs.append(ref)

        ref_audio_list = converted_refs

        # Generate speech using XTTS voice cloning with multiple references
        model.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=ref_audio_list,
            language="en",  # XTTS uses "en" as default
        )

        return {
            "success": True,
            "path": output_path,
            "type": "coqui_xtts",
            "references": [os.path.basename(r) for r in ref_audio_list],
            "num_references": len(ref_audio_list),
        }

    except Exception as e:
        raise HTTPException(status_code=500, text=f"Generation failed: {str(e)}")


# Training Data Management
def scan_itawit_training_data():
    """Scan and register Itawit training data from segments directory."""
    segments_dir = "/Users/master/vbox/voicebox/scripts/training_data/itawit/segments"
    itawit_dir = "/Users/master/vbox/voicebox/scripts/training_data/itawit"

    if not os.path.exists(segments_dir):
        return 0

    # Get transcripts from the JW toolkit manifest
    transcripts = {}
    manifest_file = (
        "/Users/master/Downloads/jw_qwen_itv_toolkit/manifests/filtered_itv.jsonl"
    )
    if os.path.exists(manifest_file):
        with open(manifest_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    audio_file = os.path.basename(data.get("audio", ""))
                    text = data.get("text", "")
                    if audio_file and text:
                        transcripts[audio_file] = text
                except:
                    pass

    # Register training data in database
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    count = 0
    for f in os.listdir(segments_dir):
        if f.endswith(".wav"):
            audio_path = os.path.join(segments_dir, f)

            # Get duration
            try:
                import subprocess

                result = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        audio_path,
                    ],
                    capture_output=True,
                    text=True,
                )
                duration = float(result.stdout.strip()) if result.stdout.strip() else 0
            except:
                duration = 0

            text = transcripts.get(f, "")

            # Check if already exists
            c.execute(
                "SELECT id FROM training_data WHERE audio_path = ?", (audio_path,)
            )
            if not c.fetchone():
                c.execute(
                    """INSERT INTO training_data (language, audio_path, transcribe_text, duration, source)
                       VALUES (?, ?, ?, ?, ?)""",
                    ("itw", audio_path, text, duration, "jw_toolkit"),
                )
                count += 1

    conn.commit()
    conn.close()

    print(f"Registered {count} Itawit training samples")
    return count


@app.post("/training/scan")
async def scan_training_data():
    """Scan directories for training data and register in database."""
    count = scan_itawit_training_data()
    return {"success": True, "samples_added": count}


@app.get("/training/data")
async def get_training_data(language: str = "itw"):
    """Get training data for a language."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """SELECT id, audio_path, transcribe_text, duration, source, is_used 
           FROM training_data WHERE language = ? ORDER BY id""",
        (language,),
    )
    rows = c.fetchall()
    conn.close()

    return {
        "training_data": [
            {
                "id": r[0],
                "audio_path": r[1],
                "text": r[2],
                "duration": r[3],
                "source": r[4],
                "is_used": bool(r[5]),
            }
            for r in rows
        ]
    }


@app.get("/training/stats")
async def get_training_stats():
    """Get training statistics."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Count by language
    c.execute("""SELECT language, COUNT(*), SUM(duration), SUM(is_used) 
                FROM training_data GROUP BY language""")
    lang_stats = c.fetchall()

    # Model stats
    c.execute(
        "SELECT name, training_epochs, num_samples, status FROM itawit_models ORDER BY id"
    )
    model_stats = c.fetchall()

    conn.close()

    return {
        "languages": [
            {
                "language": r[0],
                "total_samples": r[1],
                "total_duration": r[2] or 0,
                "used_samples": r[3] or 0,
            }
            for r in lang_stats
        ],
        "models": [
            {
                "name": r[0],
                "epochs": r[1],
                "samples": r[2],
                "status": r[3],
            }
            for r in model_stats
        ],
    }


@app.post("/training/generate")
async def generate_training_data(
    num_samples: int = Form(10),
    text: str = Form(""),
):
    """Generate synthetic training data using Itawit voice cloning."""
    if not text:
        # Default Itawit phrases from lexicon
        text_samples = [
            "Ma-ngo! Mabbalat.",
            "Jehova i Dios.",
            "Kunnasi ka? Napia nak.",
            "Anna yo ngahan mu?",
            "Ay-ayatan ta ka.",
            "Ti biblian ti kasisirinan na.",
            "Nakakkasta nga Balita nga Naggafu kan Dios!",
            "Mappagilammu ket importanti para ittayo.",
        ]
    else:
        text_samples = [text]

    model = get_coqui_model()
    if not model:
        raise HTTPException(status_code=500, text="Coqui XTTS not available")

    # Get Itawit segments as reference
    segments_dir = "/Users/master/vbox/voicebox/scripts/training_data/itawit/segments"
    ref_files = sorted([f for f in os.listdir(segments_dir) if f.endswith(".wav")])[:10]
    ref_audio_list = [os.path.join(segments_dir, f) for f in ref_files]

    if not ref_audio_list:
        raise HTTPException(status_code=400, text="No reference audio found")

    generated = []
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    for i in range(num_samples):
        text_to_generate = text_samples[i % len(text_samples)]
        output_path = f"/tmp/itawit_train_{uuid.uuid4().hex[:8]}.wav"

        try:
            model.tts_to_file(
                text=text_to_generate,
                file_path=output_path,
                speaker_wav=ref_audio_list,
            )

            # Get duration
            try:
                import subprocess

                result = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        output_path,
                    ],
                    capture_output=True,
                    text=True,
                )
                duration = float(result.stdout.strip()) if result.stdout.strip() else 0
            except:
                duration = 0

            # Save to training data
            c.execute(
                """INSERT INTO training_data (language, audio_path, text, duration, source)
                   VALUES (?, ?, ?, ?, ?)""",
                ("itw", output_path, text_to_generate, duration, "generated"),
            )

            generated.append(
                {
                    "text": text_to_generate,
                    "path": output_path,
                    "duration": duration,
                }
            )

        except Exception as e:
            print(f"Error generating sample {i}: {e}")

    conn.commit()
    conn.close()

    return {
        "success": True,
        "generated": generated,
        "total": len(generated),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
