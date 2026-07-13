import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

from ffmpeg_utils import find_ffprobe

# whisper-1 is the only OpenAI speech model that returns per-segment
# timestamps (via verbose_json / timestamp_granularities), which subtitles
# require. gpt-4o-transcribe / gpt-4o-mini-transcribe only return plain text,
# so do not "upgrade" this model unless the API gains segment timestamps.
TRANSCRIBE_MODEL = "whisper-1"
# How many chunks to transcribe concurrently. Each chunk is an independent
# request, so a small pool speeds up multi-chunk audio without risking rate
# limits.
MAX_TRANSCRIBE_WORKERS = 4

_HALLUCINATION_PATTERNS = [
    "untertitel",
    "amara.org",
    "amara.ord",
    "untertitelung",
    "copyright",
    "vielen dank",
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "subtitles by",
    "sous-titres",
]


def _is_hallucination(text: str) -> bool:
    """Detect likely Whisper hallucination segments."""
    text = text.strip()
    if not text:
        return True

    # Strip whitespace and punctuation to get core content
    core = re.sub(r'[\s\.,!?\-;:\'\"]+', '', text)

    # Only punctuation/whitespace, or single repeated character (e.g. "T T T T")
    if len(set(core.lower())) <= 1:
        return True

    # Known hallucination phrases
    lower = text.lower()
    for phrase in _HALLUCINATION_PATTERNS:
        if phrase in lower:
            return True

    return False


def _get_audio_duration(path: str) -> float:
    """Return the duration of an audio file in seconds using ffprobe."""
    result = subprocess.run(
        [
            find_ffprobe(),
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def _normalize_chunks(chunks: list) -> list[dict]:
    """Accept either legacy ``list[str]`` paths or ``list[dict]`` chunks.

    Every chunk is returned as ``{"path": str, "offset": float}``. When an offset
    is missing (legacy plain-path input), it is derived by accumulating the
    measured durations of the preceding chunks.
    """
    normalized: list[dict] = []
    running_offset = 0.0
    for chunk in chunks:
        if isinstance(chunk, dict):
            path = chunk["path"]
            offset = chunk.get("offset")
            if offset is None:
                offset = running_offset
        else:
            path = chunk
            offset = running_offset
        offset = float(offset)
        normalized.append({"path": path, "offset": offset})
        running_offset = offset + _get_audio_duration(path)
    return normalized


def _transcribe_chunk(chunk: dict, client: OpenAI) -> list[dict]:
    """Transcribe a single chunk and return its segments on the original timeline."""
    offset = chunk["offset"]
    with open(chunk["path"], "rb") as f:
        response = client.audio.transcriptions.create(
            model=TRANSCRIBE_MODEL,
            file=f,
            language="de",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    segments = []
    for segment in response.segments or []:
        text = segment.text.strip()
        if _is_hallucination(text):
            continue
        segments.append({
            "start": segment.start + offset,
            "end": segment.end + offset,
            "text": text,
        })
    return segments


def transcribe_audio(
    audio_chunks: list,
    client: OpenAI,
    max_workers: int = MAX_TRANSCRIBE_WORKERS,
) -> dict:
    """Transcribe audio chunks using the Whisper API. Returns segments + full text.

    ``audio_chunks`` is a list of ``{"path", "offset"}`` dicts (as produced by
    ``extract_audio_chunks``); plain path strings are also accepted for
    backwards compatibility. Each chunk's Whisper timestamps are anchored to the
    chunk's ``offset`` in the original audio, so long pauses between chunks never
    shift subtitle timing. A failed chunk is skipped with a warning rather than
    aborting the whole transcription.
    """
    chunks = _normalize_chunks(audio_chunks)
    results: list[list[dict]] = [[] for _ in chunks]

    def run(index: int) -> None:
        try:
            results[index] = _transcribe_chunk(chunks[index], client)
        except Exception as exc:  # noqa: BLE001 - keep going on partial failure
            print(f"    Warning: chunk {index + 1}/{len(chunks)} failed to "
                  f"transcribe ({exc}); skipping.")

    if max_workers and len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(chunks))) as pool:
            futures = [pool.submit(run, i) for i in range(len(chunks))]
            for future in as_completed(futures):
                future.result()  # run() swallows errors; this just surfaces bugs
    else:
        for i in range(len(chunks)):
            run(i)

    all_segments = [seg for chunk_segs in results for seg in chunk_segs]
    # Chunks are in timeline order, but sort defensively so timestamps are always
    # monotonically increasing even if padding makes regions touch.
    all_segments.sort(key=lambda s: (s["start"], s["end"]))

    full_text = " ".join(s["text"] for s in all_segments)
    return {"segments": all_segments, "text": full_text, "language": "de"}


def format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timecode HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt(segments: list[dict], output_path: str):
    """Write segments to an SRT file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}\n")
            f.write(f"{seg['text']}\n\n")


def translate_segments(segments: list[dict], client: OpenAI) -> list[dict]:
    """Translate German segments to English using GPT-4o."""
    translated = []

    # Batch segments for efficiency (translate in groups of 20)
    batch_size = 20
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i+batch_size]

        # Build numbered list for translation
        lines = "\n".join(f"{j+1}. {s['text']}" for j, s in enumerate(batch))

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a translator. Translate each numbered German line to English. Keep the same numbering. Only output the translations, one per line, with the same numbering format."},
                {"role": "user", "content": lines}
            ],
            temperature=0.3
        )

        translations = response.choices[0].message.content.strip().split("\n")

        for j, seg in enumerate(batch):
            # Extract translation text (remove numbering)
            if j < len(translations):
                trans_text = translations[j].strip()
                # Remove leading number and period/dot
                if trans_text and trans_text[0].isdigit():
                    parts = trans_text.split(".", 1)
                    if len(parts) > 1:
                        trans_text = parts[1].strip()
            else:
                trans_text = seg["text"]  # fallback to original

            translated.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": trans_text
            })

    return translated


def create_subtitles(audio_chunks: list, video_path: str, client: OpenAI) -> tuple[str, str, str]:
    """Main function: transcribe, create German SRT, translate, create English SRT.
    Returns (german_srt_path, english_srt_path, full_transcript_text)."""

    video_dir = os.path.dirname(os.path.abspath(video_path))
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    print("  Transcribing audio with Whisper API...")
    result = transcribe_audio(audio_chunks, client)

    german_srt = os.path.join(video_dir, f"{video_name}.de.srt")
    print(f"  Writing German subtitles: {german_srt}")
    generate_srt(result["segments"], german_srt)

    print("  Translating subtitles to English with GPT-4o...")
    translated = translate_segments(result["segments"], client)

    english_srt = os.path.join(video_dir, f"{video_name}.en.srt")
    print(f"  Writing English subtitles: {english_srt}")
    generate_srt(translated, english_srt)

    return german_srt, english_srt, result["text"]
