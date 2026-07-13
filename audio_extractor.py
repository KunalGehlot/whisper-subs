"""
audio_extractor.py

Extracts audio from MP4 files using ffmpeg via subprocess.

Splits audio into chunks at long pauses (silence) so that:
  * No single chunk exceeds the Whisper API 25 MB upload limit.
  * Whisper never has to transcribe across a long silence. Long silences
    corrupt Whisper's internal timestamps and trigger repetition/hallucination,
    which breaks the timing of everything that follows the pause. By cutting the
    audio at silence and anchoring each chunk to its known start time, subtitle
    timing stays accurate regardless of how long the speaker pauses.
"""

import os
import re
import subprocess
import tempfile
import math
from pathlib import Path

from ffmpeg_utils import find_ffmpeg, find_ffprobe

# --- Silence-splitting tuning ---------------------------------------------
# Audio quieter than this (in dBFS) is treated as silence.
SILENCE_NOISE_DB = -30
# Only pauses at least this long trigger a split. Shorter pauses are natural
# speech and are left inside a chunk; longer pauses are where Whisper's
# timestamps drift, and are also natural subtitle boundaries.
MIN_SILENCE_SEC = 2.0
# Keep a little audio on each side of a cut so word onsets aren't clipped.
CHUNK_PAD_SEC = 0.25
# Hard cap on chunk length. Bounds memory/latency and keeps every chunk far
# below the 25 MB upload limit (~6 MB at 16 kHz mono MP3).
MAX_CHUNK_SEC = 1200
# Stay under the Whisper API's 25 MB hard limit.
MAX_SIZE_MB = 24


def _get_file_size_mb(path: str) -> float:
    """Return file size in megabytes."""
    return os.path.getsize(path) / (1024 * 1024)


def _get_media_duration_seconds(media_path: str) -> float:
    """Use ffprobe to get a media file's duration in seconds."""
    result = subprocess.run(
        [
            find_ffprobe(),
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            media_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


# Backwards-compatible alias (older name referred specifically to video).
_get_video_duration_seconds = _get_media_duration_seconds


_FLOAT_RE = r"(-?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)"


def _detect_silences(
    audio_path: str,
    total_duration: float,
    noise_db: int = SILENCE_NOISE_DB,
    min_silence: float = MIN_SILENCE_SEC,
) -> list[tuple[float, float]]:
    """Return (start, end) intervals of silence at least ``min_silence`` seconds long.

    Uses ffmpeg's ``silencedetect`` filter, which reports its findings on stderr.
    """
    proc = subprocess.run(
        [
            find_ffmpeg(),
            "-hide_banner", "-nostats",
            "-i", audio_path,
            "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )
    log = proc.stderr or ""
    starts = [float(x) for x in re.findall(rf"silence_start:\s*{_FLOAT_RE}", log)]
    ends = [float(x) for x in re.findall(rf"silence_end:\s*{_FLOAT_RE}", log)]

    # If the file ends during a silence, ffmpeg may report the start but no end.
    if len(starts) == len(ends) + 1:
        ends.append(total_duration)

    return list(zip(starts, ends))


def _speech_regions(
    duration: float,
    silences: list[tuple[float, float]],
    pad: float = CHUNK_PAD_SEC,
) -> list[tuple[float, float]]:
    """Return the speech regions (complement of the silences) as (start, end) spans.

    Long silences are removed entirely so they are never sent to the API. Each
    region is padded slightly and clamped to the file bounds.
    """
    regions: list[tuple[float, float]] = []
    pos = 0.0
    for s, e in sorted(silences):
        if s > pos:
            regions.append((pos, s))
        pos = max(pos, e)
    if pos < duration:
        regions.append((pos, duration))

    padded: list[tuple[float, float]] = []
    for start, end in regions:
        start = max(0.0, start - pad)
        end = min(duration, end + pad)
        if end - start > 0.1:  # drop degenerate slivers
            padded.append((start, end))
    return padded


def _cap_regions(
    regions: list[tuple[float, float]],
    max_chunk_dur: float = MAX_CHUNK_SEC,
) -> list[tuple[float, float]]:
    """Split any region longer than ``max_chunk_dur`` into equal time slices."""
    capped: list[tuple[float, float]] = []
    for start, end in regions:
        span = end - start
        if span <= max_chunk_dur:
            capped.append((start, end))
            continue
        n = math.ceil(span / max_chunk_dur)
        step = span / n
        for k in range(n):
            cs = start + k * step
            ce = end if k == n - 1 else start + (k + 1) * step
            capped.append((cs, ce))
    return capped


def _extract_region(src: str, start: float, duration_sec: float, out_path: str) -> None:
    """Extract [start, start+duration_sec] of ``src`` into a 16 kHz mono MP3."""
    subprocess.run(
        [
            find_ffmpeg(),
            "-y",
            "-ss", f"{start:.3f}",
            "-t", f"{duration_sec:.3f}",
            "-i", src,
            "-vn",
            "-acodec", "libmp3lame",
            "-ar", "16000",
            "-ac", "1",
            "-q:a", "5",
            out_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )


def _split_by_size(
    chunk_path: str,
    base_offset: float,
    tmp_dir: str,
    idx: int,
    max_size_mb: float = MAX_SIZE_MB,
) -> list[dict]:
    """Fallback: split an over-sized chunk into equal time slices under the limit.

    Only reached for pathological inputs (a very long unbroken stretch of loud
    audio); the duration cap normally keeps chunks well under the size limit.
    """
    size = _get_file_size_mb(chunk_path)
    dur = _get_media_duration_seconds(chunk_path)
    n = max(2, math.ceil(size / max_size_mb))
    sub = dur / n
    out: list[dict] = []
    for k in range(n):
        sub_path = os.path.join(tmp_dir, f"chunk_{idx:03d}_{k:02d}.mp3")
        _extract_region(chunk_path, k * sub, sub, sub_path)
        if os.path.getsize(sub_path) > 0:
            out.append({"path": sub_path, "offset": base_offset + k * sub})
    try:
        os.remove(chunk_path)
    except OSError:
        pass
    return out


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
        [find_ffprobe(), "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    if not probe.stdout.strip():
        raise RuntimeError(f"No audio stream found in: {video_path}")

    tmp_dir = tempfile.mkdtemp(prefix="whisper_audio_")
    output_path = os.path.join(tmp_dir, "audio.mp3")

    subprocess.run(
        [
            find_ffmpeg(),
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


def extract_audio_chunks(video_path: str, max_size_mb: int = MAX_SIZE_MB) -> list[dict]:
    """
    Extract audio from a video and split it into speech chunks at long pauses.

    The audio is cut at silences at least ``MIN_SILENCE_SEC`` long. Long silences
    are dropped entirely, so Whisper only ever sees contiguous speech (plus short
    natural pauses) and its timestamps stay accurate. Each returned chunk records
    the offset of its start in the original timeline so the transcriber can place
    every segment on the real clock, no matter how long the speaker paused.

    Any region still longer than ``MAX_CHUNK_SEC`` (or, defensively, larger than
    ``max_size_mb``) is further split by time.

    Args:
        video_path:   Path to the input video file.
        max_size_mb:  Maximum size in MB for each chunk (default 24, under the
                      Whisper API's 25 MB hard limit).

    Returns:
        Ordered list of chunk dicts, each ``{"path": str, "offset": float}`` where
        ``offset`` is the chunk's start time (seconds) in the original audio.

    Raises:
        FileNotFoundError: If the video file does not exist.
        subprocess.CalledProcessError: If ffmpeg or ffprobe fails.
    """
    video_path = str(Path(video_path).resolve())
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Extract the full audio track once, then slice it locally.
    full_audio_path = extract_audio(video_path)
    duration = _get_media_duration_seconds(full_audio_path)

    silences = _detect_silences(full_audio_path, total_duration=duration)
    regions = _speech_regions(duration, silences)
    if not regions:
        # Detection found nothing (e.g. music bed above the noise floor, or an
        # entirely silent track) -- fall back to transcribing the whole file.
        regions = [(0.0, duration)]
    regions = _cap_regions(regions)

    # Fast path: one region spanning (essentially) the whole file, already small
    # enough. Avoid a needless re-encode and just hand back the full audio.
    if (
        len(regions) == 1
        and regions[0][0] <= CHUNK_PAD_SEC + 0.05
        and duration - regions[0][1] <= CHUNK_PAD_SEC + 0.05
        and _get_file_size_mb(full_audio_path) <= max_size_mb
    ):
        return [{"path": full_audio_path, "offset": 0.0}]

    tmp_dir = tempfile.mkdtemp(prefix="whisper_chunks_")
    chunks: list[dict] = []
    for i, (start, end) in enumerate(regions):
        out_path = os.path.join(tmp_dir, f"chunk_{i:03d}.mp3")
        _extract_region(full_audio_path, start, end - start, out_path)
        if os.path.getsize(out_path) == 0:
            continue
        if _get_file_size_mb(out_path) > max_size_mb:
            chunks.extend(_split_by_size(out_path, start, tmp_dir, i, max_size_mb))
        else:
            chunks.append({"path": out_path, "offset": start})

    if not chunks:
        # Nothing usable was extracted; keep the full audio as a last resort.
        return [{"path": full_audio_path, "offset": 0.0}]

    # We have standalone chunks now; the full-audio temp file is no longer needed.
    cleanup([full_audio_path])
    return chunks


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
