"""
ffmpeg_utils.py

Locate ffmpeg and ffprobe binaries, searching the project directory first
for a bundled copy before falling back to the system PATH.
"""

import platform
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent

_cached_ffmpeg: str | None = None
_cached_ffprobe: str | None = None


def _find_binary(name: str) -> str:
    """Search for a binary in project-local ffmpeg folders, then fall back to PATH."""
    suffix = ".exe" if platform.system() == "Windows" else ""
    target = name + suffix

    # Look for directories matching ffmpeg*/ that contain bin/<target>
    for folder in sorted(_PROJECT_ROOT.glob("ffmpeg*/")):
        candidate = folder / "bin" / target
        if candidate.is_file():
            return str(candidate)

    # Fall back to bare name (relies on PATH)
    return name


def find_ffmpeg() -> str:
    """Return the path to the ffmpeg binary (cached after first call)."""
    global _cached_ffmpeg
    if _cached_ffmpeg is None:
        _cached_ffmpeg = _find_binary("ffmpeg")
    return _cached_ffmpeg


def find_ffprobe() -> str:
    """Return the path to the ffprobe binary (cached after first call)."""
    global _cached_ffprobe
    if _cached_ffprobe is None:
        _cached_ffprobe = _find_binary("ffprobe")
    return _cached_ffprobe
