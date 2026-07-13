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
from path_utils import clean_path_arg
from ui import create_reporter

SUPPORTED_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm")


def resolve_video_path(raw: str) -> str:
    """Resolve the CLI video argument to an absolute path, or exit with guidance.

    Handles the common Windows failure where a quoted/dragged path (or one with
    trailing whitespace) reaches ``sys.argv`` still decorated, which otherwise
    makes a perfectly valid file look "not found".
    """
    cleaned = clean_path_arg(raw)
    video_path = os.path.abspath(cleaned)

    if not os.path.isfile(video_path):
        print(f"Error: File not found: {video_path}", file=sys.stderr)
        if cleaned != raw:
            # The argument arrived decorated (quotes/whitespace); show what we used.
            print(f"  (interpreted argument {raw!r} as {cleaned!r})", file=sys.stderr)
        print(f"  Current directory: {os.getcwd()}", file=sys.stderr)
        print("  Tip: on Windows, wrap paths with spaces in double quotes, e.g.", file=sys.stderr)
        print('       python process_video.py "C:\\Users\\me\\My Videos\\clip.mp4"', file=sys.stderr)
        sys.exit(1)

    if not video_path.lower().endswith(SUPPORTED_EXTENSIONS):
        print(f"Error: Unsupported file format. Supported: {', '.join(SUPPORTED_EXTENSIONS)}", file=sys.stderr)
        sys.exit(1)

    return video_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate German/English subtitles and a key-points report from a German-language MP4 video."
    )
    parser.add_argument("video", help="Path to the MP4 video file")
    parser.add_argument("--api-key", help="OpenAI API key (overrides OPENAI_API_KEY env var)")
    args = parser.parse_args()

    video_path = resolve_video_path(args.video)

    # Load API key
    load_dotenv()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: No OpenAI API key provided. Set OPENAI_API_KEY or use --api-key.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    reporter = create_reporter()
    reporter.header(
        "🎬  Whisper Subs",
        f"{os.path.basename(video_path)}\n{os.path.dirname(video_path)}",
    )

    audio_chunks: list = []
    ok = False
    try:
        with reporter:
            # Extract audio, splitting at long pauses so silences never corrupt
            # Whisper's timestamps.
            audio_chunks = extract_audio_chunks(video_path, reporter=reporter)
            # Transcribe -> German SRT -> translate -> English SRT.
            german_srt, english_srt, transcript = create_subtitles(
                audio_chunks, video_path, client, reporter=reporter
            )
            # Summarize the transcript into a report.
            report_path = generate_report(transcript, video_path, client, reporter=reporter)
        ok = True
    except KeyboardInterrupt:
        reporter.warn("Interrupted.")
    except Exception as exc:  # noqa: BLE001 - present failures cleanly, not as a traceback
        reporter.warn(f"Failed: {exc}")
    finally:
        cleanup([chunk["path"] for chunk in audio_chunks])

    if not ok:
        sys.exit(1)

    reporter.summary(
        "Done — generated files",
        [
            ("German subtitles", german_srt),
            ("English subtitles", english_srt),
            ("Analysis report", report_path),
        ],
    )
    reporter.note("View in VLC (auto-detects the .de.srt / .en.srt files):")
    reporter.note(f'  vlc "{video_path}"')


if __name__ == "__main__":
    main()
