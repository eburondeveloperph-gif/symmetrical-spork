#!/usr/bin/env python3
"""
Eburon TTS Training - HuggingFace Datasets + MLX/Unsloth
Train Dutch Flemish and Tagalog TTS models locally on Apple Silicon
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
from datasets import load_dataset, Audio
import soundfile as sf

LANGUAGE_CONFIG = {
    "nl": {
        "name": "Dutch (Flemish)",
        "dataset_name": "mozilla-foundation/common_voice_13_0",
        "config": "nl",
    },
    "tl": {
        "name": "Tagalog",
        "dataset_name": "mozilla-foundation/common_voice_13_0",
        "config": "tl",
    },
    "fy": {
        "name": "Fries",
        "dataset_name": "mozilla-foundation/common_voice_13_0",
        "config": "fy-NL",
    },
}


def download_dataset(language: str, max_samples: int = 1000):
    """Download dataset from HuggingFace."""
    config = LANGUAGE_CONFIG[language]
    print(f"\n📥 Downloading {config['name']} dataset...")

    try:
        ds = load_dataset(
            config["dataset_name"],
            config["config"],
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        samples = []
        for i, example in enumerate(ds):
            if i >= max_samples:
                break

            audio = example["audio"]
            if isinstance(audio, dict) and "array" in audio:
                samples.append(
                    {
                        "audio": audio["array"],
                        "text": example["sentence"],
                        "sample_rate": audio.get("sampling_rate", 32000),
                    }
                )

            if (i + 1) % 100 == 0:
                print(f"  Downloaded {i + 1}/{max_samples} samples...")

        print(f"✓ Downloaded {len(samples)} samples")
        return samples

    except Exception as e:
        print(f"Error: {e}")
        return None


def prepare_audio(audio_array, target_sr=24000, target_length=None):
    """Prepare audio for TTS training."""
    import librosa

    # Resample
    if target_sr != audio_array.sample_rate:
        audio_array = librosa.resample(
            audio_array, orig_sr=audio_array.sample_rate, target_sr=target_sr
        )

    # Normalize
    audio_array = audio_array / np.abs(audio_array).max() * 0.9

    return audio_array


def prepare_training_data(samples, output_dir: str):
    """Save training data."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save JSON metadata
    metadata = []
    for i, sample in enumerate(samples):
        audio = sample["audio"]
        if hasattr(audio, "sample_rate"):
            sr = audio.sample_rate
            arr = audio.array if hasattr(audio, "array") else audio
        else:
            sr = sample.get("sample_rate", 24000)
            arr = audio

        # Save audio file
        audio_path = output_path / f"audio_{i:05d}.wav"
        sf.write(audio_path, arr, sr)

        metadata.append({"audio_file": str(audio_path), "text": sample["text"]})

    # Save metadata
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✓ Saved {len(metadata)} samples to {output_dir}")
    return metadata


def train_with_unsloth(dataset_path: str, output_model: str, language: str):
    """Train TTS model using Unsloth."""
    print(f"\n🎓 Training {language} TTS model...")

    # Check if mlx-audio supports training
    try:
        from unsloth import FastLanguageModel
        import torch

        print("Loading base model...")
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
            bias="none",
            task_type="CAUSAL_LM",
        )

        print("✓ Model loaded with LoRA")
        print(f"\n📝 To complete training:")
        print(f"  1. Prepare your dataset in {dataset_path}")
        print(f"  2. Use a training loop with your data")
        print(f"  3. Save with: model.save_pretrained('{output_model}')")

        return model, tokenizer

    except Exception as e:
        print(f"Error: {e}")
        return None, None


def load_and_test_model(model_path: str, text: str):
    """Load trained model and test."""
    try:
        from mlx_audio.tts import load

        print(f"\n🧪 Testing model from {model_path}...")
        model = load(model_path)

        # Generate
        result = list(model.generate(text=text))[-1]

        output_file = "/tmp/test_output.wav"
        sf.write(output_file, result.audio, result.sample_rate)

        print(f"✓ Generated: {output_file}")
        print(f"   Duration: {result.audio_duration:.2f}s")

        return output_file

    except Exception as e:
        print(f"Error testing model: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Eburon TTS Training")
    parser.add_argument(
        "--language",
        "-l",
        choices=["nl", "tl", "fy"],
        default="nl",
        help="Language to train",
    )
    parser.add_argument(
        "--max-samples",
        "-n",
        type=int,
        default=500,
        help="Number of samples to download",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./training_data",
        help="Output directory for training data",
    )
    parser.add_argument(
        "--train", action="store_true", help="Start training after preparing data"
    )
    parser.add_argument("--test", type=str, help="Test a trained model with given text")
    parser.add_argument(
        "--model-path", type=str, help="Path to trained model for testing"
    )

    args = parser.parse_args()

    print("=" * 50)
    print("🎙️ Eburon TTS Training - HuggingFace + MLX")
    print("=" * 50)

    # Test existing model
    if args.test and args.model_path:
        load_and_test_model(args.model_path, args.test)
        return

    # Download dataset
    samples = download_dataset(args.language, args.max_samples)

    if samples is None:
        print("Failed to download dataset")
        return

    # Prepare training data
    prepare_training_data(samples, args.output_dir)

    # Start training
    if args.train:
        train_with_unsloth(args.output_dir, f"./models/{args.language}", args.language)

    print("\n✅ Done!")
    print(f"\nTo test with your trained model:")
    print(f"  python train_hf.py --test 'Your text' --model-path /path/to/model")


if __name__ == "__main__":
    main()
