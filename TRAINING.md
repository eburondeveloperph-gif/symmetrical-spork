# Eburon TTS Voice Training Guide

## Overview

Eburon TTS supports native languages out of the box. For better Dutch Flemish and Tagalog support, you can fine-tune the model with custom datasets.

## Quick Start - Download Pre-trained Models

We provide pre-trained LoRA adapters for Dutch Flemish and Tagalog:

### Dutch (Flemish)
```bash
# Download Dutch Flemish adapter
huggingface-cli download eburon/tts-nl-flemish-lora --local-dir ./models/dutch_flemish
```

### Tagalog
```bash
# Download Tagalog adapter  
huggingface-cli download eburon/tts-tl-tagalog-lora --local-dir ./models/tagalog
```

## Training from Scratch

### Dataset Requirements

For training a custom voice, you need:

1. **Audio files** - High quality WAV (24kHz+)
2. **Transcripts** - Matching text for each audio file
3. **Minimum** - 30 minutes of audio recommended
4. **Format** - Clean recordings, no background noise

### Dataset Format

Create a CSV file with two columns:

```csv
audio_path,text
/path/to/audio1.wav,Hello world
/path/to/audio2.wav,Good morning
/path/to/audio3.wav,How are you today
```

### Training Script

```python
from unsloth import FastLanguageModel
import torch

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)

# Configure LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# Train with your dataset
# ... training code ...
```

### Training Parameters

| Parameter | Recommended |
|-----------|-------------|
| Learning Rate | 2e-4 |
| Batch Size | 4-8 |
| Epochs | 3-10 |
| Warmup Steps | 100 |

## Dutch Flemish Dataset Sources

### Public Datasets

1. **Common Voice Dutch**
   - https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0-nl
   - ~50 hours of Dutch speech

2. **Dutch Speech Corpus**
   - https://huggingface.co/datasets/JorisK/dutch-speech
   - Various Dutch speakers

### Tagalog Dataset Sources

1. **Common Voice Tagalog**
   - https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0_tl
   - Tagalog speech data

2. **Filipino Speech Corpus**
   - https://huggingface.co/datasets/ai4bharat/ai4bharat-10k_tl

## Recording Your Own Voice

### Equipment Needed
- Quality microphone
- Quiet environment
- Audio interface (optional)

### Recording Tips
- Speak clearly at natural pace
- Vary pitch and tone
- Record 30+ minutes minimum
- Include various sentence types

### Audio Processing
```bash
# Normalize audio
ffmpeg -i input.wav -af "loudnorm=I=-16:TP=-1.5:LRA=11" output.wav

# Convert to 24kHz
ffmpeg -i input.wav -ar 24000 output.wav

# Trim silence
ffmpeg -i input.wav -af "silenceremove=start_threshold=-50dB:stop_threshold=-50dB:start_silence=0.1:stop_silence=0.1" output.wav
```

## Using Trained Models

After training, load your model:

```python
from mlx_audio.tts import load

# Load with custom adapter
model = load("your-trained-model")
result = list(model.generate(text="Hallo!", language="dutch"))[-1]
```

## Support

For issues and questions:
- GitHub: https://github.com/eburondeveloperph-gif/symmetrical-spork
- Discord: Join our community
