import os
import sys
import unittest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCleanPathArg(unittest.TestCase):
    """The Windows CLI path fix: strip decoration that breaks os.path.isfile."""

    def test_plain_path_unchanged(self):
        from path_utils import clean_path_arg
        self.assertEqual(clean_path_arg("video.mp4"), "video.mp4")
        self.assertEqual(clean_path_arg(r"C:\clips\a.mp4"), r"C:\clips\a.mp4")

    def test_surrounding_double_quotes_stripped(self):
        # The classic drag-and-drop / PowerShell case that made abspath prepend
        # the cwd and report "file not found".
        from path_utils import clean_path_arg
        self.assertEqual(clean_path_arg('"C:\\My Videos\\clip.mp4"'), "C:\\My Videos\\clip.mp4")

    def test_surrounding_single_quotes_stripped(self):
        from path_utils import clean_path_arg
        self.assertEqual(clean_path_arg("'/home/me/clip.mp4'"), "/home/me/clip.mp4")

    def test_double_wrapped_quotes_stripped(self):
        from path_utils import clean_path_arg
        self.assertEqual(clean_path_arg('""C:\\a.mp4""'), "C:\\a.mp4")

    def test_leading_and_trailing_whitespace_stripped(self):
        from path_utils import clean_path_arg
        self.assertEqual(clean_path_arg("  video.mp4\n"), "video.mp4")
        self.assertEqual(clean_path_arg(' "video.mp4" '), "video.mp4")

    def test_tilde_expanded(self):
        from path_utils import clean_path_arg
        result = clean_path_arg("~/clip.mp4")
        self.assertFalse(result.startswith("~"))
        self.assertTrue(result.endswith("clip.mp4"))

    def test_inner_quotes_preserved(self):
        # Only *surrounding* quotes are stripped; a quote inside the name stays.
        from path_utils import clean_path_arg
        self.assertEqual(clean_path_arg('my"weird.mp4'), 'my"weird.mp4')

    def test_empty_input(self):
        from path_utils import clean_path_arg
        self.assertEqual(clean_path_arg(""), "")


if __name__ == "__main__":
    unittest.main()
