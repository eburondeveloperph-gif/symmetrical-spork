#!/usr/bin/env python3
"""
Itawit Voice Cloning using Coqui XTTS v2
==========================================
This script creates native Itawit TTS using voice cloning from reference audio.
"""

import os
import sys
import argparse
import uuid
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try importing TTS - handle torchcodec import issue
try:
    import torch

    os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "0"
    from TTS.api import TTS

    COQUI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Coqui TTS not fully available: {e}")
    COQUI_AVAILABLE = False


class ItawitVoiceCloner:
    """Voice cloning for Itawit using Coqui XTTS"""

    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path
        self.xtts_model = None
        self.itawit_ref_audio = None

    def load_model(self, force_download=False):
        """Load XTTS v2 model"""
        if not COQUI_AVAILABLE:
            print("Error: Coqui TTS not available")
            return False

        try:
            print("Loading Coqui XTTS v2 model...")
            # Use XTTS v2 for best voice cloning
            self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            print("XTTS v2 model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def set_reference_audio(self, audio_path):
        """Set the reference audio for voice cloning"""
        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found: {audio_path}")
            return False

        self.itawit_ref_audio = audio_path
        print(f"Reference audio set: {audio_path}")
        return True

    def speak(self, text, output_path=None, language="it"):
        """Generate speech using voice cloning"""
        if not self.model:
            print("Error: Model not loaded")
            return None

        if not self.itawit_ref_audio:
            print("Error: Reference audio not set")
            return None

        if not output_path:
            output_path = f"/tmp/itawit_xtts_{uuid.uuid4().hex[:8]}.wav"

        try:
            print(f"Generating Itawit speech: {text}")
            print(f"  Reference: {self.itawit_ref_audio}")
            print(f"  Output: {output_path}")

            # XTTS voice cloning - using reference audio for voice cloning
            # Use the correct API with speaker_wav parameter
            self.model.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=self.itawit_ref_audio,
                language=language,
            )

            print(f"Generated: {output_path}")
            return output_path

        except Exception as e:
            print(f"Error generating speech: {e}")
            import traceback

            traceback.print_exc()
            return None


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Itawit Voice Cloning using Coqui XTTS"
    )
    parser.add_argument("--text", "-t", type=str, help="Text to speak")
    parser.add_argument(
        "--audio", "-a", type=str, default=None, help="Reference audio file"
    )
    parser.add_argument(
        "--list", "-l", action="store_true", help="List available reference audio"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output file path"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="it",
        help="Language code (default: it for Italian - we'll map to Itawit)",
    )

    args = parser.parse_args()

    # Itawit training data directory
    itawit_dir = "/Users/master/vbox/voicebox/scripts/training_data/itawit"

    if args.list:
        print("Available Itawit reference audio:")
        if os.path.exists(itawit_dir):
            for f in sorted(os.listdir(itawit_dir)):
                if f.endswith((".wav", ".mp3", ".flac")):
                    print(f"  {f}")
        return

    # Initialize voice cloner
    cloner = ItawitVoiceCloner()

    # Load model
    if not cloner.load_model():
        print("Failed to load XTTS model")
        return

    # Set reference audio
    if args.audio:
        audio_path = args.audio
    else:
        # Find first available audio
        audio_path = os.path.join(itawit_dir, "kt_ITW_01_ref.wav")
        if not os.path.exists(audio_path):
            # Try to find any audio file
            for f in os.listdir(itawit_dir):
                if f.endswith(".wav"):
                    audio_path = os.path.join(itawit_dir, f)
                    break

    if not os.path.exists(audio_path):
        print(f"Error: No reference audio found in {itawit_dir}")
        return

    cloner.set_reference_audio(audio_path)

    # Generate speech
    if args.text:
        output = cloner.speak(args.text, args.output, args.lang)
        if output:
            print(f"\n✓ Speech generated: {output}")
            # Play if possible
            try:
                import subprocess

                subprocess.run(["afplay", output], check=True)
            except:
                pass
    else:
        # Interactive mode
        print("\nItawit Voice Cloner - Interactive Mode")
        print("=" * 40)
        print(f"Reference audio: {audio_path}")
        print("Type 'quit' to exit\n")

        while True:
            try:
                text = input("Itawit > ").strip()
                if text.lower() in ("quit", "exit", "q"):
                    break
                if text:
                    cloner.speak(text, args.output)
            except KeyboardInterrupt:
                break
            except EOFError:
                break


if __name__ == "__main__":
    main()
