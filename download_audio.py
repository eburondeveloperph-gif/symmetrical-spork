#!/usr/bin/env python3
"""
Eburon TTS Dataset Downloader
Download audio from YouTube for TTS training
"""

import os
import json
import subprocess
import argparse
from pathlib import Path
import random

# Sample YouTube channels with Tagalog content
TAGALOG_SOURCES = [
    # News channels
    {"url": "https://www.youtube.com/@ABSCBNNews", "name": "ABSCBN News"},
    {"url": "https://www.youtube.com/@GMANews", "name": "GMA News"},
    {"url": "https://www.youtube.com/@TV5Manila", "name": "TV5"},
    # Educational
    {"url": "https://www.youtube.com/@DepEdPH", "name": "DepEd"},
    # Entertainment
    {
        "url": "https://www.youtube.com/@PhilippineCelebrities",
        "name": "Philippine Celebs",
    },
]

# Sample Dutch sources
DUTCH_SOURCES = [
    {"url": "https://www.youtube.com/@nos", "name": "NOS News"},
    {"url": "https://www.youtube.com/@RTLnl", "name": "RTL Netherlands"},
    {"url": "https://www.youtube.com/@NPO1", "name": "NPO 1"},
]


def check_yt_dlp():
    """Check if yt-dlp is installed."""
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True)
        return True
    except FileNotFoundError:
        return False


def download_audio(url: str, output_dir: str, max_duration: int = 60):
    """Download audio from YouTube video."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = output_path / "%(title)s.%(ext)s"

    cmd = [
        "yt-dlp",
        "-x",  # Extract audio
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",  # Best quality
        "--download-sections",
        f"*-{max_duration}",  # First N seconds
        "-o",
        str(filename),
        "--skip-download",
        "--write-info-json",
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print(f"✓ Found: {url}")
            return True
        else:
            print(f"✗ Failed: {url}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def download_playlist(url: str, output_dir: str, max_videos: int = 10):
    """Download videos from playlist."""
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "-o",
        f"{output_dir}/%(title)s.%(ext)s",
        "--playlist-items",
        f"1:{max_videos}",
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False


def get_video_list(url: str, max_results: int = 10):
    """Get list of videos from channel/playlist."""
    cmd = ["yt-dlp", "--playlist-items", f"1:{max_results}", "--print", "%(url)s", url]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            return result.stdout.strip().split("\n")
        return []
    except Exception as e:
        print(f"Error getting videos: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Download audio for TTS training")
    parser.add_argument(
        "--language",
        "-l",
        choices=["tl", "nl"],
        default="tl",
        help="Language: tl (Tagalog) or nl (Dutch)",
    )
    parser.add_argument(
        "--output-dir", "-o", default="./audio_data", help="Output directory"
    )
    parser.add_argument(
        "--max-videos", "-n", type=int, default=5, help="Maximum videos to download"
    )

    args = parser.parse_args()

    sources = TAGALOG_SOURCES if args.language == "tl" else DUTCH_SOURCES

    print(f"📥 Downloading {args.language} audio for TTS training...")
    print(f"Output: {args.output_dir}")
    print()

    if not check_yt_dlp():
        print("Installing yt-dlp...")
        subprocess.run(["pip", "install", "yt-dlp"], check=True)

    total_downloaded = 0

    for source in sources:
        print(f"\n📺 {source['name']}...")
        videos = get_video_list(source["url"], args.max_videos)

        for video_url in videos[:3]:  # Download first 3 from each
            if video_url:
                print(f"  Downloading: {video_url[:50]}...")
                if download_audio(video_url, args.output_dir):
                    total_downloaded += 1

    print(f"\n✓ Done! Downloaded {total_downloaded} videos")
    print(f"Files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
