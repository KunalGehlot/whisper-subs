"""
audio_extractor.py

Extracts audio from MP4 files using ffmpeg via subprocess.
Handles the Whisper API 25MB file size limit by splitting large audio into chunks.
"""

import os
import subprocess
import tempfile
import math
from pathlib import Path


def _get_file_size_mb(path: str) -> float:
    """Return file size in megabytes."""
    return os.path.getsize(path) / (1024 * 1024)


def _get_video_duration_seconds(video_path: str) -> float:
    """Use ffprobe to get video duration in seconds."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def extract_audio(video_path: str) -> str:
    """
    Extract audio from an MP4 file to a temporary MP3 file.

    If the resulting MP3 is under 25MB, returns a single path.
    For files that might exceed the limit, prefer extract_audio_chunks().

    Args:
        video_path: Absolute path to the input MP4 file.

    Returns:
        Path to the extracted temporary MP3 file.

    Raises:
        FileNotFoundError: If the video file does not exist.
        subprocess.CalledProcessError: If ffmpeg fails.
    """
    video_path = str(Path(video_path).resolve())
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Check that the file has an audio stream
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    if not probe.stdout.strip():
        raise RuntimeError(f"No audio stream found in: {video_path}")

    tmp_dir = tempfile.mkdtemp(prefix="whisper_audio_")
    output_path = os.path.join(tmp_dir, "audio.mp3")

    subprocess.run(
        [
            "ffmpeg",
            "-y",                  # overwrite without prompting
            "-i", video_path,
            "-vn",                 # no video
            "-acodec", "libmp3lame",
            "-ar", "16000",        # 16 kHz – sufficient for speech
            "-ac", "1",            # mono
            "-q:a", "5",           # variable bitrate quality (smaller file)
            output_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return output_path


def extract_audio_chunks(video_path: str, max_size_mb: int = 24) -> list[str]:
    """
    Extract audio from an MP4 and split into chunks that each stay under max_size_mb.

    Uses ffmpeg's segment feature to produce sequentially numbered MP3 files,
    then filters out any empty/missing segments.

    Args:
        video_path:   Absolute path to the input MP4 file.
        max_size_mb:  Maximum size in MB for each chunk (default 24 to stay under
                      the Whisper API's 25 MB hard limit).

    Returns:
        Sorted list of paths to the temporary MP3 chunk files.

    Raises:
        FileNotFoundError: If the video file does not exist.
        subprocess.CalledProcessError: If ffmpeg or ffprobe fails.
    """
    video_path = str(Path(video_path).resolve())
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # First, extract the full audio to measure its size
    full_audio_path = extract_audio(video_path)
    full_size_mb = _get_file_size_mb(full_audio_path)

    # If it already fits, return it as a single-element list
    if full_size_mb <= max_size_mb:
        return [full_audio_path]

    # Calculate how many chunks we need
    num_chunks = math.ceil(full_size_mb / max_size_mb)
    duration = _get_video_duration_seconds(video_path)
    segment_duration = math.ceil(duration / num_chunks)

    tmp_dir = tempfile.mkdtemp(prefix="whisper_chunks_")
    segment_pattern = os.path.join(tmp_dir, "chunk_%03d.mp3")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "libmp3lame",
            "-ar", "16000",
            "-ac", "1",
            "-q:a", "5",
            "-f", "segment",
            "-segment_time", str(segment_duration),
            "-reset_timestamps", "1",
            segment_pattern,
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    # Clean up the unsplit full audio now that we have chunks
    cleanup([full_audio_path])

    # Collect and sort chunk paths
    chunk_paths = sorted(
        str(p)
        for p in Path(tmp_dir).glob("chunk_*.mp3")
        if p.stat().st_size > 0
    )

    return chunk_paths


def cleanup(paths: list[str]) -> None:
    """
    Remove temporary audio files and their parent temp directories if empty.

    Args:
        paths: List of file paths returned by extract_audio or extract_audio_chunks.
    """
    dirs_to_check: set[str] = set()
    for path in paths:
        try:
            if os.path.isfile(path):
                dirs_to_check.add(os.path.dirname(path))
                os.remove(path)
        except OSError:
            pass  # Best-effort cleanup

    for d in dirs_to_check:
        try:
            if os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
        except OSError:
            pass
