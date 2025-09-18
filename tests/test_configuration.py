import unittest

from civitAI_Model_downloader import DownloaderConfig
from gui_app import DownloaderGUI


class DownloaderConfigTests(unittest.TestCase):
    def test_usernames_are_trimmed_and_filtered(self):
        config = DownloaderConfig(usernames=[" alice ", "", "bob"])
        self.assertEqual(config.usernames, ["alice", "bob"])

    def test_missing_usernames_raise_error(self):
        with self.assertRaises(ValueError):
            DownloaderConfig(usernames=["   "])

    def test_none_download_type_defaults_to_all(self):
        config = DownloaderConfig(usernames=["alice"], download_type=None)
        self.assertEqual(config.download_type, "All")


class DownloaderGUITests(unittest.TestCase):
    def test_parse_usernames_handles_commas_and_newlines(self):
        raw = "alice,\nbob\n\ncharlie"
        parsed = DownloaderGUI._parse_usernames(raw)
        self.assertEqual(parsed, ["alice", "bob", "charlie"])


if __name__ == "__main__":
    unittest.main()
