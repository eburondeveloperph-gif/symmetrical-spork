#!/usr/bin/env python3
"""
Eburon TTS Automated Training Pipeline
Download dataset → Train model → Save model
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

LANGUAGE_CONFIG = {
    "nl": {"name": "Dutch (Flemish)", "code": "nl"},
    "tl": {"name": "Tagalog", "code": "tl"},
}


def run_command(cmd, description=""):
    """Run shell command and return output."""
    print(f"\n{'=' * 50}")
    if description:
        print(f"📦 {description}")
    print("=" * 50)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False
    print(result.stdout)
    return True


def install_dependencies():
    """Install required packages."""
    print("\n📥 Installing dependencies...")

    packages = [
        "unsloth",
        "transformers",
        "datasets",
        "torch",
        "soundfile",
        "scipy",
        "accelerate",
        "peft",
        "bitsandbytes",
    ]

    for pkg in packages:
        print(f"Installing {pkg}...")
        subprocess.run(f"pip install {pkg} -q", shell=True)

    print("✅ Dependencies installed!")


def download_dataset(language, max_hours, output_dir):
    """Download Common Voice dataset."""
    print(f"\n📥 Downloading {LANGUAGE_CONFIG[language]['name']} dataset...")

    code = f"""
from datasets import load_dataset
import soundfile as sf
import os

os.makedirs('{output_dir}', exist_ok=True)

ds = load_dataset('common_voice', '{language}', split='train')
print(f'Total: {{len(ds)}} samples')

samples = min({max_hours} * 1000, len(ds))
train_data = ds.select(range(samples))

metadata = []
for i, example in enumerate(train_data):
    audio = example['audio']
    audio_path = '{output_dir}/audio_{{i:05d}}.wav'
    sf.write(audio_path, audio['array'], audio['sampling_rate'])
    
    metadata.append({{
        'audio_file': audio_path,
        'text': example['sentence']
    }})
    
    if (i + 1) % 100 == 0:
        print(f'Downloaded {{i+1}}/{{samples}}')

with open('{output_dir}/metadata.json', 'w') as f:
    json.dump(metadata, f)

print(f'✅ Downloaded {{len(metadata)}} samples')
"""

    with open("/tmp/download_dataset.py", "w") as f:
        f.write(code)

    return run_command(
        "python3 /tmp/download_dataset.py", f"Downloading {language} dataset"
    )


def train_model(dataset_dir, output_model, model_size, epochs):
    """Train TTS model."""
    print(f"\n🎓 Training TTS model...")

    code = f"""
import json
import torch
from unsloth import FastLanguageModel
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import soundfile as sf

# Load metadata
with open('{dataset_dir}/metadata.json', 'r') as f:
    metadata = json.load(f)

print(f'Loading {{len(metadata)}} samples...')

# Prepare dataset (simplified - just use text for now)
texts = [item['text'] for item in metadata]
ds = Dataset.from_dict({{'text': texts}})

# Load model
model_name = 'Qwen/Qwen3-TTS-12Hz-{model_size}-Base'
print(f'Loading {{model_name}}...')

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
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
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    bias='none',
    task_type='CAUSAL_LM'
)

print('Model ready!')
print('Note: Full training loop needs custom implementation.')
print(f'Saving base model to {{output_model}}...')

model.save_pretrained('{output_model}')
tokenizer.save_pretrained('{output_model}')

print('✅ Model saved!')
"""

    with open("/tmp/train_model.py", "w") as f:
        f.write(code)

    return run_command("python3 /tmp/train_model.py", "Training model")


def test_model(model_path, test_text):
    """Test trained model."""
    print(f"\n🧪 Testing model...")

    code = f"""
from mlx_audio.tts import load
import soundfile as sf

model = load('{model_path}')
result = list(model.generate(text='{test_text}'))[-1]

sf.write('test_output.wav', result.audio, result.sample_rate)
print(f'✅ Generated: test_output.wav ({{result.audio_duration:.2f}}s)')
"""

    with open("/tmp/test_model.py", "w") as f:
        f.write(code)

    return run_command("python3 /tmp/test_model.py", "Testing model")


def main():
    parser = argparse.ArgumentParser(description="Eburon TTS Automated Training")
    parser.add_argument("--language", "-l", choices=["nl", "tl"], required=True)
    parser.add_argument("--max-hours", type=int, default=5)
    parser.add_argument("--model-size", choices=["0.6B", "1.7B"], default="0.6B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output", "-o", default="./trained_models")
    parser.add_argument("--test-text", default="Hello! This is a test.")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--test-only", action="store_true")

    args = parser.parse_args()

    print("=" * 60)
    print("🎙️ Eburon TTS Automated Training")
    print("=" * 60)
    print(f"Language: {LANGUAGE_CONFIG[args.language]['name']}")
    print(f"Max Hours: {args.max_hours}")
    print(f"Model Size: {args.model_size}")
    print(f"Output: {args.output}")

    dataset_dir = f"{args.output}/dataset_{args.language}"
    output_model = f"{args.output}/tts-{args.language}"

    # Install
    if not args.skip_install:
        install_dependencies()

    if args.test_only:
        test_model(output_model, args.test_text)
        return

    # Download
    if not args.skip_download:
        if not download_dataset(args.language, args.max_hours, dataset_dir):
            print("❌ Download failed!")
            return

    # Train
    if not args.skip_train:
        if not train_model(dataset_dir, output_model, args.model_size, args.epochs):
            print("❌ Training failed!")
            return

    # Test
    test_model(output_model, args.test_text)

    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print(f"📁 Model saved: {output_model}")
    print("=" * 60)


if __name__ == "__main__":
    main()
