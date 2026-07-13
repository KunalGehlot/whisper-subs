import os
import sys
import shutil
import subprocess
import tempfile
import unittest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "test_german.ogg")


# --- Test helpers ----------------------------------------------------------

def _ffmpeg_available() -> bool:
    from ffmpeg_utils import find_ffmpeg
    return shutil.which(find_ffmpeg()) is not None or os.path.isfile(find_ffmpeg())


def _synth_audio(path: str, spec: list) -> None:
    """Synthesize a mono 16 kHz audio file from a spec of tone/silence segments.

    spec items: ("tone", frequency_hz, duration_s) or ("silence", duration_s).
    """
    from ffmpeg_utils import find_ffmpeg

    inputs: list[str] = []
    for i, item in enumerate(spec):
        if item[0] == "tone":
            _, freq, dur = item
            inputs += ["-f", "lavfi", "-i",
                       f"sine=frequency={freq}:sample_rate=16000:duration={dur}"]
        else:
            _, dur = item
            inputs += ["-f", "lavfi", "-i", f"anullsrc=r=16000:cl=mono:d={dur}"]

    concat_in = "".join(f"[{i}:a]" for i in range(len(spec)))
    filt = f"{concat_in}concat=n={len(spec)}:v=0:a=1[out]"
    cmd = ([find_ffmpeg(), "-y"] + inputs +
           ["-filter_complex", filt, "-map", "[out]", "-ar", "16000", "-ac", "1", path])
    subprocess.run(cmd, capture_output=True, text=True, check=True)


class _FakeSegment:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


class _FakeResponse:
    def __init__(self, segments):
        self.segments = segments


class _FakeTranscriptions:
    """Returns canned segments keyed by the chunk file's basename."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def create(self, model=None, file=None, **kwargs):
        key = os.path.basename(file.name)
        self.calls.append(key)
        return _FakeResponse(self.mapping.get(key, []))


class _FakeClient:
    def __init__(self, mapping):
        self.audio = type("A", (), {"transcriptions": _FakeTranscriptions(mapping)})()

class TestHallucinationFilter(unittest.TestCase):
    """Test the hallucination detection filter."""

    def test_repeated_characters_detected(self):
        from transcriber import _is_hallucination
        self.assertTrue(_is_hallucination("T T T T T"))
        self.assertTrue(_is_hallucination("TTTTT"))
        self.assertTrue(_is_hallucination("..."))

    def test_known_hallucination_phrases_detected(self):
        from transcriber import _is_hallucination
        self.assertTrue(_is_hallucination("Untertitel der Amara.org community"))
        self.assertTrue(_is_hallucination("Untertitelung des ZDF"))
        self.assertTrue(_is_hallucination("Copyright 2024"))
        self.assertTrue(_is_hallucination("Vielen Dank für's Zuschauen"))

    def test_legitimate_text_passes(self):
        from transcriber import _is_hallucination
        self.assertFalse(_is_hallucination("Hallo, wie geht es Ihnen?"))
        self.assertFalse(_is_hallucination("Montag, Dienstag, Mittwoch"))
        self.assertFalse(_is_hallucination("eins, zwei, drei, vier, fünf"))
        self.assertFalse(_is_hallucination("Die Katze sitzt auf dem Tisch"))

    def test_empty_text_detected(self):
        from transcriber import _is_hallucination
        self.assertTrue(_is_hallucination(""))
        self.assertTrue(_is_hallucination("   "))


class TestFfmpegDiscovery(unittest.TestCase):
    """Test ffmpeg binary discovery."""

    def test_find_ffmpeg_returns_string(self):
        from ffmpeg_utils import find_ffmpeg
        result = find_ffmpeg()
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_find_ffprobe_returns_string(self):
        from ffmpeg_utils import find_ffprobe
        result = find_ffprobe()
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)


class TestAudioDuration(unittest.TestCase):
    """Test audio duration measurement."""

    def test_fixture_exists(self):
        self.assertTrue(os.path.exists(FIXTURE_PATH), f"Test fixture not found: {FIXTURE_PATH}")

    def test_audio_duration(self):
        from transcriber import _get_audio_duration
        duration = _get_audio_duration(FIXTURE_PATH)
        # Should be approximately 25 seconds
        self.assertGreater(duration, 20.0)
        self.assertLess(duration, 35.0)


@unittest.skipUnless(os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY not set")
class TestTranscription(unittest.TestCase):
    """Test actual transcription (requires API key)."""

    def test_transcribe_german_audio(self):
        from openai import OpenAI
        from transcriber import transcribe_audio

        client = OpenAI()
        result = transcribe_audio([FIXTURE_PATH], client)

        # Should have segments
        self.assertGreater(len(result["segments"]), 0)

        # Timestamps should be monotonically increasing
        for i in range(1, len(result["segments"])):
            self.assertGreaterEqual(
                result["segments"][i]["start"],
                result["segments"][i-1]["start"],
                "Timestamps should be monotonically increasing"
            )

        # Should contain some expected German words
        text_lower = result["text"].lower()
        has_expected = any(word in text_lower for word in [
            "montag", "dienstag", "mittwoch", "donnerstag", "freitag",
            "eins", "zwei", "drei", "vier", "fünf",
            "rot", "grün", "gelb", "blau", "weiß", "schwarz",
            "test", "1", "2", "3"
        ])
        self.assertTrue(has_expected, f"Transcript should contain expected German words, got: {result['text']}")

        # No hallucinations should have made it through
        from transcriber import _is_hallucination
        for seg in result["segments"]:
            self.assertFalse(
                _is_hallucination(seg["text"]),
                f"Hallucination detected in output: {seg['text']}"
            )


class TestSpeechRegions(unittest.TestCase):
    """Pure logic: computing speech regions from detected silences."""

    def test_no_silence_returns_whole_file(self):
        from audio_extractor import _speech_regions
        regions = _speech_regions(30.0, [])
        self.assertEqual(len(regions), 1)
        self.assertAlmostEqual(regions[0][0], 0.0, places=3)
        self.assertAlmostEqual(regions[0][1], 30.0, places=3)

    def test_long_silence_splits_into_two_regions(self):
        from audio_extractor import _speech_regions
        # speech 0-5, silence 5-11, speech 11-16
        regions = _speech_regions(16.0, [(5.0, 11.0)], pad=0.25)
        self.assertEqual(len(regions), 2)
        # First region ends shortly after speech, second begins shortly before it.
        self.assertLess(regions[0][1], regions[1][0])
        self.assertAlmostEqual(regions[0][0], 0.0, places=3)
        self.assertAlmostEqual(regions[1][1], 16.0, places=3)
        # The long silence itself is excluded from both regions.
        self.assertLessEqual(regions[0][1], 5.5)
        self.assertGreaterEqual(regions[1][0], 10.5)

    def test_leading_silence_dropped(self):
        from audio_extractor import _speech_regions
        # File starts with a long silence 0-8, then speech to 20.
        regions = _speech_regions(20.0, [(0.0, 8.0)], pad=0.25)
        self.assertEqual(len(regions), 1)
        self.assertGreater(regions[0][0], 7.0)  # first region starts after silence

    def test_trailing_sliver_dropped(self):
        from audio_extractor import _speech_regions
        # A tiny speech island shorter than the drop threshold is discarded.
        regions = _speech_regions(10.0, [(0.0, 4.95), (5.0, 10.0)], pad=0.0)
        self.assertEqual(regions, [])


class TestCapRegions(unittest.TestCase):
    def test_short_region_unchanged(self):
        from audio_extractor import _cap_regions
        self.assertEqual(_cap_regions([(0.0, 100.0)], max_chunk_dur=1200), [(0.0, 100.0)])

    def test_long_region_split_evenly_and_contiguously(self):
        from audio_extractor import _cap_regions
        out = _cap_regions([(0.0, 2500.0)], max_chunk_dur=1000)
        self.assertEqual(len(out), 3)  # ceil(2500/1000)
        # Contiguous, covers the whole span, none exceeds the cap.
        self.assertAlmostEqual(out[0][0], 0.0)
        self.assertAlmostEqual(out[-1][1], 2500.0)
        for i in range(1, len(out)):
            self.assertAlmostEqual(out[i][0], out[i - 1][1])
        for s, e in out:
            self.assertLessEqual(e - s, 1000.0 + 1e-6)


class TestTranscribeOffsets(unittest.TestCase):
    """The core fix: each chunk's timestamps are anchored to its real offset.

    Runs fully offline with a fake OpenAI client, so no API key is required.
    """

    def setUp(self):
        if not _ffmpeg_available():
            self.skipTest("ffmpeg not available")
        self.tmp = tempfile.mkdtemp(prefix="offset_test_")
        self.a = os.path.join(self.tmp, "chunk_000.mp3")
        self.b = os.path.join(self.tmp, "chunk_001.mp3")
        _synth_audio(self.a, [("tone", 440, 3)])
        _synth_audio(self.b, [("tone", 660, 3)])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_offsets_applied_to_each_chunk(self):
        from transcriber import transcribe_audio
        mapping = {
            "chunk_000.mp3": [_FakeSegment(0.5, 1.5, "Hallo Welt")],
            # Second chunk starts at t=100s in the real timeline; Whisper reports
            # times relative to the chunk (0.2-1.0), which must be shifted by 100.
            "chunk_001.mp3": [_FakeSegment(0.2, 1.0, "Guten Tag")],
        }
        client = _FakeClient(mapping)
        result = transcribe_audio(
            [{"path": self.a, "offset": 0.0}, {"path": self.b, "offset": 100.0}],
            client,
            max_workers=1,
        )
        segs = result["segments"]
        self.assertEqual(len(segs), 2)
        self.assertAlmostEqual(segs[0]["start"], 0.5, places=3)
        self.assertAlmostEqual(segs[0]["end"], 1.5, places=3)
        self.assertAlmostEqual(segs[1]["start"], 100.2, places=3)
        self.assertAlmostEqual(segs[1]["end"], 101.0, places=3)
        # A long pause must NOT compress later timestamps back toward the first
        # chunk -- this is the exact regression being guarded against.
        self.assertGreater(segs[1]["start"], 99.0)

    def test_hallucinations_filtered_per_chunk(self):
        from transcriber import transcribe_audio
        mapping = {
            "chunk_000.mp3": [
                _FakeSegment(0.0, 1.0, "Untertitel der Amara.org community"),
                _FakeSegment(1.0, 2.0, "Echte Sprache hier"),
            ],
            "chunk_001.mp3": [_FakeSegment(0.0, 1.0, "T T T T T")],
        }
        client = _FakeClient(mapping)
        result = transcribe_audio(
            [{"path": self.a, "offset": 0.0}, {"path": self.b, "offset": 50.0}],
            client,
            max_workers=1,
        )
        texts = [s["text"] for s in result["segments"]]
        self.assertEqual(texts, ["Echte Sprache hier"])

    def test_legacy_list_of_paths_accumulates_offsets(self):
        from transcriber import transcribe_audio
        mapping = {
            "chunk_000.mp3": [_FakeSegment(0.5, 1.0, "eins")],
            "chunk_001.mp3": [_FakeSegment(0.5, 1.0, "zwei")],
        }
        client = _FakeClient(mapping)
        # Legacy plain-path input: second chunk's offset is derived from the
        # measured duration (~3s) of the first.
        result = transcribe_audio([self.a, self.b], client, max_workers=1)
        segs = result["segments"]
        self.assertEqual(len(segs), 2)
        self.assertAlmostEqual(segs[0]["start"], 0.5, places=2)
        self.assertGreater(segs[1]["start"], 3.0)  # shifted past the first chunk

    def test_one_failing_chunk_does_not_abort(self):
        from transcriber import transcribe_audio

        class Boom(_FakeTranscriptions):
            def create(self, model=None, file=None, **kwargs):
                key = os.path.basename(file.name)
                if key == "chunk_000.mp3":
                    raise RuntimeError("simulated API failure")
                return super().create(model=model, file=file, **kwargs)

        client = _FakeClient({"chunk_001.mp3": [_FakeSegment(0.0, 1.0, "überlebt")]})
        client.audio.transcriptions = Boom({"chunk_001.mp3": [_FakeSegment(0.0, 1.0, "überlebt")]})
        result = transcribe_audio(
            [{"path": self.a, "offset": 0.0}, {"path": self.b, "offset": 10.0}],
            client,
            max_workers=1,
        )
        self.assertEqual([s["text"] for s in result["segments"]], ["überlebt"])


@unittest.skipUnless(_ffmpeg_available(), "ffmpeg not available")
class TestSilenceSplitting(unittest.TestCase):
    """Integration: real ffmpeg silence detection and chunk extraction."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="silence_test_")
        self._created_chunks = []

    def tearDown(self):
        from audio_extractor import cleanup
        cleanup(self._created_chunks)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_detect_long_silence(self):
        from audio_extractor import _detect_silences, _get_media_duration_seconds
        path = os.path.join(self.tmp, "pause.wav")
        _synth_audio(path, [("tone", 440, 5), ("silence", 6), ("tone", 660, 5)])
        dur = _get_media_duration_seconds(path)
        silences = _detect_silences(path, total_duration=dur, min_silence=2.0)
        self.assertEqual(len(silences), 1)
        s, e = silences[0]
        self.assertAlmostEqual(s, 5.0, delta=0.3)
        self.assertAlmostEqual(e, 11.0, delta=0.3)

    def test_long_pause_produces_two_chunks_with_correct_offsets(self):
        from audio_extractor import extract_audio_chunks, _get_media_duration_seconds
        # 5s speech, 6s pause, 5s speech -> must split into two chunks.
        path = os.path.join(self.tmp, "video_like.wav")
        _synth_audio(path, [("tone", 440, 5), ("silence", 6), ("tone", 660, 5)])

        chunks = extract_audio_chunks(path)
        self._created_chunks = [c["path"] for c in chunks]

        self.assertEqual(len(chunks), 2)
        # First chunk anchored at the start.
        self.assertLessEqual(chunks[0]["offset"], 0.3)
        # Second chunk anchored near the end of the pause (~11s), NOT right after
        # the first chunk -- proving the silence is accounted for in the timeline.
        self.assertGreater(chunks[1]["offset"], 9.5)
        self.assertLess(chunks[1]["offset"], 11.5)
        # Each extracted chunk is a real, non-empty, short audio file.
        for c in chunks:
            self.assertTrue(os.path.isfile(c["path"]))
            self.assertGreater(os.path.getsize(c["path"]), 0)
            self.assertLess(_get_media_duration_seconds(c["path"]), 8.0)

    def test_no_long_pause_returns_single_chunk(self):
        from audio_extractor import extract_audio_chunks
        path = os.path.join(self.tmp, "continuous.wav")
        _synth_audio(path, [("tone", 440, 8)])
        chunks = extract_audio_chunks(path)
        self._created_chunks = [c["path"] for c in chunks]
        self.assertEqual(len(chunks), 1)
        self.assertAlmostEqual(chunks[0]["offset"], 0.0, places=3)


if __name__ == "__main__":
    unittest.main()
