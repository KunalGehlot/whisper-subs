import os
import sys
import unittest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "test_german.ogg")

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


if __name__ == "__main__":
    unittest.main()
