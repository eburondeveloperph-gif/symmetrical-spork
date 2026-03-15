# Eburon TTS - Echo Model

AI Text-to-Speech service powered by Qwen3 TTS with MLX acceleration for Apple Silicon.

## Features

- 🎙️ High-quality text-to-speech generation
- ⚡ Fast inference with MLX acceleration (~3x realtime on M1/M2/M3)
- 🎨 Beautiful web interface
- 💾 WAV audio output

## Quick Start

```bash
# Install dependencies
pip install fastapi uvicorn mlx-audio soundfile

# Run the server
python eburon_tts_server.py
```

Then open http://localhost:8000

## API Usage

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello! This is Eburon TTS."}'
```

## Requirements

- Python 3.12+
- Apple Silicon Mac (M1/M2/M3)
- mlx-audio library

## License

MIT
