"""Make fake 'authors' with different word habits, formatted like Gutenberg files.

Used only by the tests. Each author has their own tilt on how often the common
words appear (their 'style'); each book also boosts a few rare words (its 'topic').
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

VOCAB = 300


def words(n: int = VOCAB):
    return ["w" + chr(97 + i // 26) + chr(97 + i % 26) for i in range(n)]


def author_distribution(rng, tilt: float = 0.6):
    base = 1.0 / np.arange(1, VOCAB + 1)
    base[:80] *= np.exp(rng.normal(0, tilt, 80))  # this author's habits with common words
    return base / base.sum()


def book_distribution(rng, p):
    q = p.copy()
    topic = rng.choice(np.arange(80, VOCAB), size=30, replace=False)
    q[topic] *= 8  # this book's topic words
    return q / q.sum()


def format_gutenberg(title: str, author: str, tokens, wid: int) -> str:
    lines, para, sent = [], [], []
    for i, tok in enumerate(tokens):
        sent.append(tok)
        if len(sent) == 14:
            para.append(" ".join(sent).capitalize() + ".")
            sent = []
        if len(para) == 4:
            lines.append(" ".join(para))
            para = []
        if i and i % 1500 == 0:
            lines.append("CHAPTER %s." % ("I" * (i // 1500)))
    if sent:
        para.append(" ".join(sent) + ".")
    if para:
        lines.append(" ".join(para))
    return (
        "The Project Gutenberg eBook of %s\n\nTitle: %s\nAuthor: %s\n\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK %s ***\n\nProduced by Test Volunteers\n\n%s\n\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK %s ***\nLicense text that must not be counted.\n"
        % (title, title, author, title.upper(), "\n\n".join(lines), title.upper())
    )


def make_corpus(root, n_authors=5, books=4, tokens_per_book=12000, seed=7):
    """Write raw books under root/raw and return a manifest dict for them."""
    rng = np.random.default_rng(seed)
    vocab = words()
    manifest = {
        "chunk_size": 1000,
        "min_chunk": 500,
        "folds": 3,
        "seed": "test",
        "cull": 0.5,
        "gate": 0.90,
        "pairs": [["Auth0", "Auth1"]],
        "authors": {},
    }
    wid = 1000
    for a in range(n_authors):
        key = "Auth%d" % a
        p = author_distribution(rng)
        works = []
        for b in range(books):
            wid += 1
            tokens = [vocab[i] for i in rng.choice(VOCAB, size=tokens_per_book, p=book_distribution(rng, p))]
            title = "Book %d of %s" % (b, key)
            raw = format_gutenberg(title, "%s, Test" % key, tokens, wid)
            path = Path(root) / "raw" / ("%d.txt" % wid)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(raw, encoding="utf-8")
            works.append({"id": wid, "expect": "book %d" % b})
        manifest["authors"][key] = {"name": key, "works": works}
    return manifest
