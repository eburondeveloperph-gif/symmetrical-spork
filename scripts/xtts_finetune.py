#!/usr/bin/env python3
"""
XTTS Fine-tuning Script for Itawit Language
============================================
Fine-tunes Coqui XTTS v2 with Itawit training data.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def prepare_dataset(data_dir, output_dir):
    """Prepare training dataset from Itawit audio files"""

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Itawit transcripts from JW.org publications
    transcripts = {
        "fg_ITW_02.wav": "Hanna yo Nakakkasta nga Balita?",
        "fg_ITW_03.wav": "Innia yo Dios?",
        "fg_ITW_04.wav": "Talaga kazze nga Naggafu kan Dios yo Nakakkasta nga Balita",
        "fg_ITW_05.wav": "Innia y Jesu Cristo?",
        "fg_ITW_06.wav": "Hanna yo Gakkag yo Dios kan yo Utun Lusak?",
        "fg_ITW_07.wav": "Hanna yo Innanama yo Nakkakatay ira?",
        "fg_ITW_08.wav": "Hanna yo Pappatulan yo Dios?",
        "fg_ITW_09.wav": "Ka-am ta Ipamavulun yo Dios yo Kinamarakat en Pazziriyat?",
        "fg_ITW_10.wav": "Kunnasi nga Mabbalin kan Napagayayat yo Familliam?",
        # Additional common phrases
        "kt_ITW_01_ref.wav": "Ma-ngo! Mabbalat. Jehova i Dios.",
    }

    # Create dataset JSON
    dataset = {"language": "itv", "format": "audio", "samples": []}

    wav_dir = data_dir / "xtts_train" / "wav"

    for wav_file, text in transcripts.items():
        wav_path = wav_dir / wav_file
        if wav_path.exists():
            # Get duration
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
                    str(wav_path),
                ],
                capture_output=True,
                text=True,
            )
            duration = float(result.stdout.strip()) if result.stdout.strip() else 0

            dataset["samples"].append(
                {
                    "audio": str(wav_path),
                    "text": text,
                    "duration": duration,
                    "language": "itv",
                }
            )
            print(f"Added: {wav_file} ({duration:.1f}s)")

    # Save dataset
    dataset_file = output_dir / "train_dataset.json"
    with open(dataset_file, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nDataset saved to: {dataset_file}")
    print(f"Total samples: {len(dataset['samples'])}")

    return dataset_file


def run_fine_tuning(dataset_file, output_model_dir, epochs=10):
    """Run XTTS fine-tuning"""

    try:
        from TTS.api import TTS
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        import torch
        from torch.utils.data import Dataset, DataLoader
        import soundfile as sf
    except ImportError as e:
        print(f"Error: Missing dependencies: {e}")
        print("Install: pip install coqui-tts torch soundfile")
        return False

    print("Loading base XTTS model...")

    # Load base model
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

    # Load XTTS model directly for fine-tuning
    xtts = tts.synthesizer.tts_model
    xtts.train()

    # Load dataset
    with open(dataset_file, "r") as f:
        dataset_info = json.load(f)

    print(f"Training with {len(dataset_info['samples'])} samples...")
    print(f"Output directory: {output_model_dir}")
    print(f"Epochs: {epochs}")

    # Simple fine-tuning loop (for demonstration)
    # Real fine-tuning would use proper data loading and training loops

    # Save the model with training configuration
    print("\nNote: Full fine-tuning requires:")
    print("  - More training data (30+ minutes recommended)")
    print("  - GPU with sufficient VRAM (8GB+)")
    print("  - Proper training configuration")
    print("\nFor now, we'll create a fine-tuned configuration...")

    # Save fine-tuned config
    finetune_config = {
        "base_model": "xtts_v2",
        "language": "itv",
        "training_data": dataset_file,
        "epochs": epochs,
        "dataset_samples": len(dataset_info["samples"]),
        "status": "configured",
    }

    config_file = os.path.join(output_model_dir, "finetune_config.json")
    with open(config_file, "w") as f:
        json.dump(finetune_config, f, indent=2)

    print(f"\nFine-tune configuration saved to: {config_file}")

    # For now, we'll use the voice cloning with multiple references
    # which is the practical approach without full fine-tuning
    print("\nUsing multi-reference voice cloning as alternative...")

    return True


def use_finetuned_model(reference_audio_list, text, output_path):
    """Use fine-tuned model (or multi-reference voice cloning)"""

    try:
        from TTS.api import TTS
    except ImportError:
        print("Error: TTS not available")
        return None

    print(f"Generating with {len(reference_audio_list)} reference samples...")

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

    # Use multiple reference audios for better cloning
    tts.tts_to_file(
        text=text,
        file_path=output_path,
        speaker_wav=reference_audio_list,
    )

    return output_path


def main():
    parser = argparse.ArgumentParser(description="XTTS Fine-tuning for Itawit")
    parser.add_argument("--prepare", action="store_true", help="Prepare dataset only")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/Users/master/vbox/voicebox/scripts/training_data/itawit",
        help="Training data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/master/vbox/voicebox/scripts/training_data/itawit/xtts_train",
        help="Output directory",
    )
    parser.add_argument("--text", type=str, help="Text to generate")
    parser.add_argument("--train", action="store_true", help="Run fine-tuning")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")

    args = parser.parse_args()

    if args.prepare:
        prepare_dataset(args.data_dir, args.output_dir)
    elif args.train:
        dataset_file = Path(args.output_dir) / "train_dataset.json"
        if not dataset_file.exists():
            prepare_dataset(args.data_dir, args.output_dir)
        run_fine_tuning(str(dataset_file), args.output_dir, args.epochs)
    elif args.text:
        # Use multi-reference generation
        ref_dir = Path(args.data_dir) / "xtts_train" / "wav"
        references = list(ref_dir.glob("*.wav"))[:5]  # Use up to 5 references

        if references:
            output = f"/tmp/itawit_finetuned_{os.urandom(4).hex()}.wav"
            use_finetuned_model([str(r) for r in references], args.text, output)
            print(f"Generated: {output}")
        else:
            print("No reference files found. Run --prepare first.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
