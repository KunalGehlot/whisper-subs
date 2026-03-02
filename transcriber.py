import os
import re
import subprocess
from openai import OpenAI

from ffmpeg_utils import find_ffprobe

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


def transcribe_audio(audio_paths: list[str], client: OpenAI) -> dict:
    """Transcribe audio files using Whisper API. Returns verbose JSON with segments."""
    all_segments = []
    offset = 0.0

    for audio_path in audio_paths:
        chunk_duration = _get_audio_duration(audio_path)

        with open(audio_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="de",
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )

        for segment in response.segments:
            text = segment.text.strip()
            if _is_hallucination(text):
                continue
            all_segments.append({
                "start": segment.start + offset,
                "end": segment.end + offset,
                "text": text
            })

        offset += chunk_duration

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


def create_subtitles(audio_paths: list[str], video_path: str, client: OpenAI) -> tuple[str, str, str]:
    """Main function: transcribe, create German SRT, translate, create English SRT.
    Returns (german_srt_path, english_srt_path, full_transcript_text)."""

    video_dir = os.path.dirname(os.path.abspath(video_path))
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    print("  Transcribing audio with Whisper API...")
    result = transcribe_audio(audio_paths, client)

    german_srt = os.path.join(video_dir, f"{video_name}.de.srt")
    print(f"  Writing German subtitles: {german_srt}")
    generate_srt(result["segments"], german_srt)

    print("  Translating subtitles to English with GPT-4o...")
    translated = translate_segments(result["segments"], client)

    english_srt = os.path.join(video_dir, f"{video_name}.en.srt")
    print(f"  Writing English subtitles: {english_srt}")
    generate_srt(translated, english_srt)

    return german_srt, english_srt, result["text"]
