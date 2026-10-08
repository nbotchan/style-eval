import unittest

from style_eval.text import (
    apply_window,
    chunk_tokens,
    clean_body,
    header_field,
    split_gutenberg,
)

RAW = """The Project Gutenberg eBook of Test

Title: A Test Book
Author: Doe, Jane

*** START OF THE PROJECT GUTENBERG EBOOK A TEST BOOK ***

Produced by Kind Volunteers

CHAPTER XII.

It was the best of times.

THE END

*** END OF THE PROJECT GUTENBERG EBOOK A TEST BOOK ***
License text.
"""


class TextTests(unittest.TestCase):
    def test_split_gutenberg_removes_license_and_reads_header(self):
        header, body = split_gutenberg(RAW)
        self.assertEqual(header_field(header, "Title"), "A Test Book")
        self.assertEqual(header_field(header, "Author"), "Doe, Jane")
        self.assertIn("best of times", body)
        self.assertNotIn("License text", body)

    def test_missing_marker_raises(self):
        with self.assertRaises(ValueError):
            split_gutenberg("no markers here")

    def test_clean_body_keeps_prose_and_drops_furniture(self):
        _, body = split_gutenberg(RAW)
        body += "\n\n[Illustration: a cat]\n\nIV.\n\n12\n\nHe said “don’t go” and left.\n"
        cleaned = clean_body(body)
        self.assertIn("It was the best of times.", cleaned)
        self.assertIn('He said "don\'t go" and left.', cleaned)
        for gone in ("CHAPTER", "THE END", "Produced by", "Illustration", "IV.", "\n12"):
            self.assertNotIn(gone, cleaned)

    def test_lowercase_words_that_look_like_numerals_are_kept(self):
        self.assertIn("did", clean_body("did\n\nI did.").split())

    def test_chunking_drops_short_tail(self):
        toks = ["x"] * 4500
        self.assertEqual([len(c) for c in chunk_tokens(toks, 2000, 1000)], [2000, 2000])
        toks = ["x"] * 5200
        self.assertEqual([len(c) for c in chunk_tokens(toks, 2000, 1000)], [2000, 2000, 1200])

    def test_apply_window(self):
        self.assertEqual(apply_window("aaa START bbb END ccc", "START", "END"), "START bbb ")
        with self.assertRaises(ValueError):
            apply_window("abc", "missing")


if __name__ == "__main__":
    unittest.main()
