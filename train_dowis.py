#!/usr/bin/env python3
"""
Eburon TTS Training - Using DOWIS Dataset
Download and train TTS with Do What I Say dataset (11 languages)
"""

import os
import json
import argparse
from pathlib import Path
import soundfile as sf


def download_dowis_dataset(language: str = None, max_samples: int = 100):
    """Download DOWIS dataset from HuggingFace."""
    from datasets import load_dataset

    print("📥 Downloading DOWIS dataset...")

    # Load full dataset
    ds = load_dataset("maikezu/dowis")

    print(f"Available columns: {ds['train'].column_names}")
    print(f"Total samples: {len(ds['train'])}")

    # Filter by language if specified
    if language:
        ds = ds.filter(lambda x: x["target_language"] == language)
        print(f"Filtered to {language}: {len(ds['train'])} samples")

    return ds


def download_and_prepare(output_dir: str, language: str = None):
    """Download and prepare dataset for training."""
    from datasets import load_dataset

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("📥 Downloading DOWIS dataset...")
    ds = load_dataset("maikezu/dowis")

    # Get samples
    samples = []

    print("Processing samples...")
    for i, example in enumerate(ds["train"]):
        if language and example.get("target_language") != language:
            continue

        # Get audio if available
        audio_data = example.get("audio")
        if audio_data:
            audio_array = audio_data["array"]
            sr = audio_data["sampling_rate"]

            # Get transcription
            text = example.get("text", example.get("transcription", ""))

            if text and len(audio_array) > 0:
                # Save audio
                audio_path = output_path / f"audio_{i:05d}.wav"
                sf.write(audio_path, audio_array, sr)

                samples.append(
                    {
                        "audio_file": str(audio_path),
                        "text": text,
                        "language": example.get(
                            "target_language", language or "unknown"
                        ),
                        "style": example.get("prompt_style", "unknown"),
                    }
                )

        if max_samples and len(samples) >= max_samples:
            break

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1} samples...")

    # Save metadata
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"✓ Saved {len(samples)} samples to {output_dir}")
    return samples


def list_available_languages():
    """List all languages in DOWIS dataset."""
    from datasets import load_dataset

    ds = load_dataset("maikezu/dowis")

    languages = set()
    for example in ds["train"]:
        if "target_language" in example:
            languages.add(example["target_language"])

    print("Available languages in DOWIS:")
    for lang in sorted(languages):
        print(f"  - {lang}")

    return languages


def main():
    parser = argparse.ArgumentParser(description="Eburon TTS - DOWIS Dataset")
    parser.add_argument("--download", action="store_true", help="Download dataset")
    parser.add_argument("--language", "-l", type=str, help="Filter by language")
    parser.add_argument(
        "--max-samples", "-n", type=int, default=100, help="Max samples to download"
    )
    parser.add_argument(
        "--output-dir", "-o", default="./dowis_data", help="Output directory"
    )
    parser.add_argument(
        "--list-languages", action="store_true", help="List available languages"
    )

    args = parser.parse_args()

    print("=" * 50)
    print("🎙️ Eburon TTS - DOWIS Dataset")
    print("=" * 50)

    if args.list_languages:
        list_available_languages()
        return

    if args.download:
        download_and_prepare(args.output_dir, args.language, args.max_samples)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
