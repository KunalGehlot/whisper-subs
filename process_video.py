#!/usr/bin/env python3
"""
Whisper Subtitle Generator & Report Tool

Processes an MP4 video with German audio to produce:
  - German subtitles (.de.srt)
  - English subtitles (.en.srt)
  - Key points report (_report.md)
"""

import sys
import os
import argparse
from dotenv import load_dotenv
from openai import OpenAI

from audio_extractor import extract_audio_chunks, cleanup
from transcriber import create_subtitles
from report_generator import generate_report


def main():
    parser = argparse.ArgumentParser(
        description="Generate German/English subtitles and a key-points report from a German-language MP4 video."
    )
    parser.add_argument("video", help="Path to the MP4 video file")
    parser.add_argument("--api-key", help="OpenAI API key (overrides OPENAI_API_KEY env var)")
    args = parser.parse_args()

    # Validate input file
    video_path = os.path.abspath(args.video)
    if not os.path.isfile(video_path):
        print(f"Error: File not found: {video_path}", file=sys.stderr)
        sys.exit(1)
    SUPPORTED_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm")
    if not video_path.lower().endswith(SUPPORTED_EXTENSIONS):
        print(f"Error: Unsupported file format. Supported: {', '.join(SUPPORTED_EXTENSIONS)}", file=sys.stderr)
        sys.exit(1)

    # Load API key
    load_dotenv()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: No OpenAI API key provided. Set OPENAI_API_KEY or use --api-key.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(video_path)}")
    print(f"{'='*60}\n")

    # Step 1: Extract audio
    print("[1/3] Extracting audio from video...")
    audio_paths = extract_audio_chunks(video_path)
    print(f"  Extracted {len(audio_paths)} audio chunk(s)\n")

    try:
        # Step 2: Transcribe and create subtitles
        print("[2/3] Creating subtitles...")
        german_srt, english_srt, transcript = create_subtitles(audio_paths, video_path, client)
        print()

        # Step 3: Generate report
        print("[3/3] Generating report...")
        report_path = generate_report(transcript, video_path, client)
        print()

    finally:
        # Clean up temp audio files
        print("Cleaning up temporary files...")
        cleanup(audio_paths)

    # Summary
    print(f"\n{'='*60}")
    print("Done! Generated files:")
    print(f"  German subtitles:  {german_srt}")
    print(f"  English subtitles: {english_srt}")
    print(f"  Analysis report:   {report_path}")
    print(f"{'='*60}")
    print(f"\nTo view with subtitles in VLC:")
    print(f"  vlc \"{video_path}\"")
    print(f"  (VLC will auto-detect the .de.srt and .en.srt files)")


if __name__ == "__main__":
    main()
