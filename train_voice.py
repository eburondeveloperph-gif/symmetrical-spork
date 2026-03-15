#!/usr/bin/env python3
"""
Eburon TTS Voice Trainer
Fine-tune TTS models using Unsloth for custom voices in Dutch Flemish and Tagalog.
"""

import os
import argparse
from unsloth import FastLanguageModel
import torch

LANGUAGE_CONFIGS = {
    "nl": {
        "name": "Dutch (Flemish)",
        "model_name": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "dataset_path": "eburon/dutch-flemish-tts",
        "output_model": "eburon/tts-nl-v1",
    },
    "tl": {
        "name": "Tagalog",
        "model_name": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "dataset_path": "eburon/tagalog-tts",
        "output_model": "eburon/tts-tl-v1",
    },
}

VOICE_CONFIGS = {
    "flemish_male": {
        "speaker_id": "male_adult_nl",
        "description": "Adult male Flemish speaker",
    },
    "flemish_female": {
        "speaker_id": "female_adult_nl",
        "description": "Adult female Flemish speaker",
    },
    "tagalog_male": {
        "speaker_id": "male_adult_tl",
        "description": "Adult male Tagalog speaker",
    },
    "tagalog_female": {
        "speaker_id": "female_adult_tl",
        "description": "Adult female Tagalog speaker",
    },
}


def load_model_and_tokenizer(model_name: str, load_in_4bit: bool = False):
    """Load model and tokenizer for fine-tuning."""
    print(f"Loading model: {model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=load_in_4bit,
    )

    return model, tokenizer


def prepare_lora_config():
    """Configure LoRA for fine-tuning."""
    return {
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def train_voice_model(
    language: str,
    voice_config: str,
    output_dir: str,
    epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    gradient_accumulation: int = 4,
):
    """Train a custom voice model."""

    if language not in LANGUAGE_CONFIGS:
        raise ValueError(
            f"Unsupported language: {language}. Choose from: {list(LANGUAGE_CONFIGS.keys())}"
        )

    if voice_config not in VOICE_CONFIGS:
        raise ValueError(
            f"Unknown voice config: {voice_config}. Choose from: {list(VOICE_CONFIGS.keys())}"
        )

    lang_config = LANGUAGE_CONFIGS[language]
    voice_info = VOICE_CONFIGS[voice_config]

    print(f"\n{'=' * 50}")
    print(f"Training {lang_config['name']} voice model")
    print(f"Voice: {voice_info['description']}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 50}\n")

    model_name = lang_config["model_name"]

    model, tokenizer = load_model_and_tokenizer(model_name)

    lora_config = prepare_lora_config()

    model = FastLanguageModel.get_peft_model(model, **lora_config)

    print("\n✅ Model prepared for training!")
    print(f"   Language: {lang_config['name']}")
    print(f"   Voice: {voice_info['description']}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")

    print("\n📝 To complete training, add your dataset and run:")
    print(f"   model.train()")
    print(f"   # Add training loop here")
    print(f"   model.save_pretrained('{output_dir}')")

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Eburon TTS Voice Trainer")
    parser.add_argument(
        "--language",
        "-l",
        choices=["nl", "tl"],
        required=True,
        help="Language: nl (Dutch/Flemish) or tl (Tagalog)",
    )
    parser.add_argument("--voice", "-v", required=True, help="Voice config name")
    parser.add_argument(
        "--output", "-o", required=True, help="Output directory for trained model"
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", "-b", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")

    args = parser.parse_args()

    train_voice_model(
        language=args.language,
        voice_config=args.voice,
        output_dir=args.output,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
