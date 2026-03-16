# Eburon TTS Training with HuggingFace Datasets

## Quick Start

```bash
# Download Dutch Flemish dataset (500 samples)
python train_hf.py -l nl -n 500 -o ./data/dutch

# Download Tagalog dataset
python train_hf.py -l tl -n 500 -o ./data/tagalog

# Start training
python train_hf.py -l nl -o ./data/dutch --train
```

## HuggingFace Datasets Used

### Dutch (Flemish)
- **Dataset:** `mozilla-foundation/common_voice_13_0`
- **Config:** `nl`
- **Hours available:** ~50 hours

### Tagalog  
- **Dataset:** `mozilla-foundation/common_voice_13_0`
- **Config:** `tl`
- **Hours available:** ~20 hours

### Fries (Flemish-related)
- **Dataset:** `mozilla-foundation/common_voice_13_0`
- **Config:** `fy-NL`

## Other Useful TTS Datasets

```python
# Multilingual TTS
from datasets import load_dataset
ds = load_dataset("facebook/fairseq/wmt19.en-de", split="train")

# VoxPopuli (multi-language)
ds = load_dataset("facebook/voxpopuli", "nl", split="train")

# CSS10 (German, Dutch, etc)
ds = load_dataset("deepmind/css10", "nl", split="train")

# LJ Speech (English)
ds = load_dataset("lj_speech", split="train")

# VCTK (English multi-speaker)
ds = load_dataset("vctk", split="train")
```

## Training with MLX on Apple Silicon

```python
from unsloth import FastLanguageModel
import torch

# Load model
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
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# Train with your dataset
# ... training loop ...

# Save
model.save_pretrained("./trained-model")
```

## Cloud Training (Google Colab)

See `eburon_tts_training.ipynb` for step-by-step Colab training.

## Model Requirements

| Model | VRAM | Training Time |
|-------|------|---------------|
| 0.6B | ~8GB | ~1 hour |
| 1.7B | ~16GB | ~2 hours |
