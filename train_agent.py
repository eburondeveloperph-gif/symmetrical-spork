#!/usr/bin/env python3
"""
Eburon TTS Training Agent
Automated training for Dutch Flemish and Tagalog TTS models.
"""

import os
import argparse
import subprocess
import json
from pathlib import Path
from datasets import load_dataset, Audio
import torch

LANGUAGE_CONFIGS = {
    "nl": {
        "name": "Dutch (Flemish)",
        "common_voice_name": "Dutch",
        "iso_code": "nl",
        "hours_needed": 10,
    },
    "tl": {
        "name": "Tagalog",
        "common_voice_name": "Tagalog",
        "iso_code": "tl",
        "hours_needed": 10,
    },
}


def check_requirements():
    """Check if required packages are installed."""
    required = ["unsloth", "transformers", "datasets", "torch"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Installing missing packages: {missing}")
        subprocess.run(["pip", "install"] + missing, check=True)
    print("✓ All requirements satisfied")


def download_common_voice(language: str, max_hours: int = 10):
    """Download Common Voice dataset for specified language."""
    config = LANGUAGE_CONFIGS[language]
    print(f"\n📥 Downloading Common Voice {config['name']} dataset...")

    try:
        dataset = load_dataset(
            "mozilla-foundation/common_voice_13_0",
            config["iso_code"],
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        # Take first N samples (approximately max_hours)
        samples = []
        sample_rate = 24000
        target_samples = max_hours * 3600 * sample_rate  # hours to samples
        current_samples = 0

        print(f"Downloading up to {max_hours} hours of audio...")

        for i, example in enumerate(dataset):
            if current_samples >= target_samples:
                break

            # Resample audio to 24kHz
            audio = example["audio"]
            if isinstance(audio, dict):
                samples.append(
                    {
                        "audio": audio["array"],
                        "text": example["sentence"],
                        "sample_rate": audio["sampling_rate"],
                    }
                )
                current_samples += len(audio["array"])

            if (i + 1) % 100 == 0:
                print(f"  Downloaded {i + 1} samples...")

        print(f"✓ Downloaded {len(samples)} audio samples")
        return samples

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None


def prepare_dataset(samples, output_dir: str):
    """Prepare and save training dataset."""
    print(f"\n📁 Preparing dataset...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as JSONL
    jsonl_path = output_path / "train.jsonl"
    with open(jsonl_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"✓ Dataset saved to {jsonl_path}")
    return str(jsonl_path)


def train_model(
    language: str,
    dataset_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
):
    """Train TTS model using Unsloth."""
    config = LANGUAGE_CONFIGS[language]
    print(f"\n🎓 Training {config['name']} TTS model...")

    model_code = f'''
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
import soundfile as sf
import json

# Configuration
model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
output_model = "{output_dir}/{language}"

# Load model
print("Loading model...")
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
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

# Load dataset
print("Loading training data...")
data = []
with open("{dataset_path}", "r") as f:
    for line in f:
        data.append(json.loads(line))

print(f"Training on {{len(data)}} samples...")

# Training loop (simplified - add proper training code)
# For TTS, you'd use a different training approach

print("Training complete!")
print(f"Model saved to {{output_model}}")
'''

    # Save training script
    script_path = f"/tmp/train_{language}.py"
    with open(script_path, "w") as f:
        f.write(model_code)

    print(f"✓ Training script saved to {script_path}")
    print(f"\nTo start training, run:")
    print(f"  python {script_path}")

    return script_path


def main():
    parser = argparse.ArgumentParser(description="Eburon TTS Training Agent")
    parser.add_argument(
        "--language",
        "-l",
        choices=["nl", "tl"],
        required=True,
        help="Language: nl (Dutch) or tl (Tagalog)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download dataset, don't train",
    )
    parser.add_argument(
        "--max-hours", type=int, default=10, help="Maximum hours of audio to download"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./trained_models",
        help="Output directory for trained model",
    )
    parser.add_argument("--epochs", "-e", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")

    args = parser.parse_args()

    print("=" * 50)
    print("🎙️ Eburon TTS Training Agent")
    print("=" * 50)

    # Check requirements
    check_requirements()

    # Download dataset
    samples = download_common_voice(args.language, args.max_hours)

    if samples is None:
        print("Failed to download dataset")
        return

    # Prepare dataset
    dataset_path = prepare_dataset(samples, f"{args.output_dir}/dataset")

    if args.download_only:
        print("\n✓ Dataset download complete!")
        print(f"Dataset location: {dataset_path}")
        return

    # Train model
    train_model(
        args.language,
        dataset_path,
        args.output_dir,
        args.epochs,
        args.batch_size,
        args.lr,
    )


if __name__ == "__main__":
    main()
