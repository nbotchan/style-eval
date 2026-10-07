import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from synthetic import make_corpus  # noqa: E402

from style_eval.corpus import build_chunks, fetch_corpus, mirror_dir, verify_header  # noqa: E402
from style_eval.text import split_gutenberg  # noqa: E402
from style_eval.trustcheck import render_report, run_trust_check  # noqa: E402

SWEEP = (20, 40, 60, 80, 100)


class TrustCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.manifest = make_corpus(cls.tmp.name)
        cls.result = run_trust_check(cls.manifest, cls.tmp.name, sweep=SWEEP, n_boot=50)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_known_answer_synthetic_authors_are_found(self):
        # Fake authors with clearly different habits must be told apart on unseen books.
        cos = self.result["methods"]["cosine"]
        self.assertGreaterEqual(cos["headline"], 0.95)
        self.assertTrue(self.result["passed"])
        self.assertEqual(cos["books_correct"], cos["books_total"])
        self.assertLessEqual(cos["ci"][0], cos["headline"])
        self.assertGreaterEqual(cos["ci"][1], cos["headline"])

    def test_books_are_held_out_not_just_chunks(self):
        self.assertEqual(self.result["folds"], 3)
        # 5 authors x 4 books x 12 chunks of 1000 words.
        self.assertEqual(self.result["n_chunks"], 5 * 4 * 12)

    def test_boilerplate_is_not_counted(self):
        chunks, missing = build_chunks(self.manifest, self.tmp.name)
        self.assertEqual(missing, [])
        flat = {t for c in chunks for t in c.tokens}
        for word in ("license", "volunteers", "chapter", "gutenberg", "produced"):
            self.assertNotIn(word, flat)

    def test_check_can_fail(self):
        # A guard against a check that always passes: give two authors a shuffled mix of
        # each other's books, so their labels carry no signal. Accuracy must drop.
        mixed = dict(self.manifest)
        a0 = self.manifest["authors"]["Auth0"]["works"]
        a1 = self.manifest["authors"]["Auth1"]["works"]
        mixed["authors"] = dict(self.manifest["authors"])
        mixed["authors"]["Auth0"] = {"name": "Auth0", "works": a0[:2] + a1[:2]}
        mixed["authors"]["Auth1"] = {"name": "Auth1", "works": a0[2:] + a1[2:]}
        res = run_trust_check(mixed, self.tmp.name, sweep=SWEEP, n_boot=20)
        self.assertLess(res["methods"]["cosine"]["per_author"]["Auth0"], 0.9)
        self.assertLess(res["methods"]["cosine"]["headline"], self.result["methods"]["cosine"]["headline"])

    def test_report_renders(self):
        report = render_report(self.result)
        self.assertIn("**PASS.**", report)
        self.assertIn("Auth3", report)
        self.assertIn("Confusion matrix", report)


class FetchTests(unittest.TestCase):
    def test_mirror_layout(self):
        self.assertEqual(mirror_dir("https://m.example/", 1342), "https://m.example/1/3/4/1342")
        self.assertEqual(mirror_dir("https://m.example", 98), "https://m.example/9/98")

    def test_header_verification(self):
        header = "Title: Pride and Prejudice\nAuthor: Austen, Jane\n"
        self.assertIsNone(verify_header(header, "Austen", "pride and prejudice"))
        self.assertIn("author", verify_header(header, "Dickens", "pride"))
        self.assertIn("title", verify_header(header, "Austen", "emma"))

    def test_fetch_from_local_mirror_accepts_right_book_and_rejects_wrong_one(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as data:
            manifest = make_corpus(src, n_authors=1, books=2)
            works = manifest["authors"]["Auth0"]["works"]
            mirror = Path(src) / "mirror"
            for work in works:
                raw = (Path(src) / "raw" / ("%d.txt" % work["id"])).read_text()
                target = Path(mirror_dir(str(mirror), work["id"])) / ("%d-0.txt" % work["id"])
                target.parent.mkdir(parents=True)
                target.write_text(raw, encoding="utf-8")
            # Ask for the right book, and for the second book under the wrong expected title.
            manifest["authors"]["Auth0"]["works"][1]["expect"] = "something else"
            summary = fetch_corpus(manifest, data, local_mirror=str(mirror), log=lambda *_: None)
            self.assertEqual(len(summary["ok"]), 1)
            self.assertEqual(len(summary["rejected"]), 1)
            self.assertTrue((Path(data) / "raw" / ("%d.txt" % works[0]["id"])).exists())
            header, _ = split_gutenberg((Path(data) / "raw" / ("%d.txt" % works[0]["id"])).read_text())
            self.assertIn("Book 0", header)


if __name__ == "__main__":
    unittest.main()
