"""The trust check: can Burrows' Delta tell the authors apart on books it has never seen?

Before any model is scored, the ruler itself has to work. This test holds out
whole books (never just chunks, because chunks from one book share characters and
topic), fits faststylometry's Burrows' Delta on the other books, and asks which
author each held-out chunk is closest to. Books are spread over `folds` rounds so
every book is held out once.

The gate comes from the spec: at least 90% top-1 in the closed set. We report the
macro average (every author counts equally) over a sweep of vocabulary sizes, so no
single setting can be cherry-picked.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np
from faststylometry import Corpus, calculate_burrows_delta

from .corpus import assign_folds, build_chunks

SWEEP = (100, 300, 500)
REPORT_N = 300  # vocabulary size used for the confusion matrix and per-book results


def _pct(x: float) -> str:
    return "%.1f%%" % (100.0 * x)


def _corpus(chunks, ids) -> Corpus:
    """One faststylometry 'book' per chunk, named "<n>" with n the chunk's global index."""
    return Corpus([chunks[i].author for i in ids], [str(i) for i in ids], [chunks[i].tokens for i in ids])


def run_trust_check(manifest: dict, data_dir, sweep=SWEEP) -> dict:
    chunks, missing = build_chunks(manifest, data_dir)
    authors = sorted({c.author for c in chunks})
    if len(authors) < 2:
        raise ValueError("need text from at least two authors; run `style-eval fetch` first")
    books: Dict[str, List[int]] = {a: sorted({c.work_id for c in chunks if c.author == a}) for a in authors}
    thin = [a for a, b in books.items() if len(b) < 2]
    if thin:
        raise ValueError("need at least 2 books per author; too few for: %s" % ", ".join(thin))
    k = min(manifest.get("folds", 3), min(len(b) for b in books.values()))

    fold_of: Dict[int, int] = {}
    for a in authors:
        fold_of.update(assign_folds(books[a], k, "%s-%s" % (manifest.get("seed", ""), a)))

    truth = np.array([authors.index(c.author) for c in chunks])
    book_of = np.array([c.work_id for c in chunks])
    fold_ids = np.array([fold_of[c.work_id] for c in chunks])

    # pred[ni, i] is the author index chosen for chunk i with sweep[ni] frequent words.
    pred = np.zeros((len(sweep), len(chunks)), dtype=int)
    for f in range(k):
        test_idx = np.where(fold_ids == f)[0]
        train_idx = np.where(fold_ids != f)[0]
        for ni, n in enumerate(sweep):
            # faststylometry stores state on the corpora, so build fresh ones per call.
            delta = calculate_burrows_delta(_corpus(chunks, train_idx), _corpus(chunks, test_idx), vocab_size=n)
            # Rows are train authors; columns are test chunks, sorted by label (not input order),
            # so map each column back to its chunk through the "<author> - <n>" label.
            nearest = delta.idxmin(axis=0)
            for label, author in nearest.items():
                pred[ni, int(label.rsplit(" - ", 1)[1])] = authors.index(author)

    report_i = int(np.argmin([abs(n - REPORT_N) for n in sweep]))
    correct = pred == truth[None, :]
    macro = np.mean([correct[:, truth == a].mean(axis=1) for a in range(len(authors))], axis=0)
    headline = float(macro.mean())

    confusion = np.zeros((len(authors), len(authors)), dtype=int)
    for t, p in zip(truth, pred[report_i]):
        confusion[t, p] += 1

    pairs = []
    for a, b in manifest.get("pairs", []):
        if a in authors and b in authors:
            ia, ib = authors.index(a), authors.index(b)
            pairs.append(
                {
                    "a": a,
                    "b": b,
                    "a_as_b": float(np.mean(pred[:, truth == ia] == ib)),
                    "b_as_a": float(np.mean(pred[:, truth == ib] == ia)),
                }
            )

    # Book-level: majority vote of a held-out book's chunks at REPORT_N.
    titles = {c.work_id: c.title for c in chunks}
    book_results, misattributed = [], []
    for wid in sorted(set(book_of.tolist())):
        sel = np.where(book_of == wid)[0]
        winner, count = Counter(pred[report_i][sel].tolist()).most_common(1)[0]
        ok = winner == truth[sel[0]]
        book_results.append(ok)
        if not ok:
            misattributed.append(
                {
                    "author": authors[truth[sel[0]]],
                    "title": titles[wid],
                    "predicted": authors[winner],
                    "share": count / len(sel),
                }
            )

    gate = manifest.get("gate", 0.9)
    return {
        "authors": authors,
        "n_chunks": len(chunks),
        "chunk_size": manifest.get("chunk_size", 2000),
        "books_per_author": {a: len(books[a]) for a in authors},
        "chunks_per_author": {a: int((truth == i).sum()) for i, a in enumerate(authors)},
        "folds": k,
        "sweep": list(sweep),
        "report_n": sweep[report_i],
        "missing": missing,
        "gate": gate,
        "macro_by_n": [float(x) for x in macro],
        "micro_by_n": [float(x) for x in correct.mean(axis=1)],
        "headline": headline,
        "per_author": {a: float(correct[:, truth == i].mean()) for i, a in enumerate(authors)},
        "confusion": confusion.tolist(),
        "pairs": pairs,
        "books_correct": int(sum(book_results)),
        "books_total": len(book_results),
        "misattributed": misattributed,
        "passed": headline >= gate,
    }


def render_report(r: dict) -> str:
    """Turn a trust-check result into a short Markdown report."""
    verdict = "PASS" if r["passed"] else "FAIL"
    micro = float(np.mean(r["micro_by_n"]))
    lines = [
        "# Style Eval trust check",
        "",
        "**%s.** Burrows' Delta (faststylometry) names the right author for %s of held-out "
        "text (gate %s; %s counting every chunk equally)."
        % (verdict, _pct(r["headline"]), _pct(r["gate"]), _pct(micro)),
        "",
        "The test: whole books are held out, Delta is fitted on the rest, and each "
        "%d-word chunk of a held-out book is assigned to the nearest of %d authors. "
        "The score is the share assigned correctly, averaged over authors (so a big "
        "author cannot carry it) and over %d to %d frequent words (so no single "
        "setting can be picked). If real text by an author is not attributed to them, "
        "the ruler is broken, not the model."
        % (r["chunk_size"], len(r["authors"]), r["sweep"][0], r["sweep"][-1]),
        "",
        "## Setup",
        "",
        "- %d authors, %d chunks, %d rounds of held-out books."
        % (len(r["authors"]), r["n_chunks"], r["folds"]),
        "",
        "## Accuracy by number of frequent words",
        "",
        "| Words | " + " | ".join(str(n) for n in r["sweep"]) + " |",
        "| --- |" + " --- |" * len(r["sweep"]),
        "| Right | " + " | ".join(_pct(x) for x in r["macro_by_n"]) + " |",
        "",
        "## Per author (averaged over the sweep)",
        "",
        "| Author | Books | Chunks | Right |",
        "| --- | --- | --- | --- |",
    ]
    for a in r["authors"]:
        lines.append(
            "| %s | %d | %d | %s |"
            % (a, r["books_per_author"][a], r["chunks_per_author"][a], _pct(r["per_author"][a]))
        )
    lines += ["", "## Where it goes wrong", ""]
    if r["pairs"]:
        lines += ["Close pairs, averaged over the sweep (how often one is mistaken for the other):", ""]
        for pr in r["pairs"]:
            lines.append(
                "- %s taken for %s: %s. %s taken for %s: %s."
                % (pr["a"], pr["b"], _pct(pr["a_as_b"]), pr["b"], pr["a"], _pct(pr["b_as_a"]))
            )
        lines.append("")
    lines += [
        "Confusion matrix at %d words (rows are the true author, columns the guess):" % r["report_n"],
        "",
        "| | " + " | ".join(r["authors"]) + " |",
        "| --- |" + " --- |" * len(r["authors"]),
    ]
    for a, row in zip(r["authors"], r["confusion"]):
        lines.append("| **%s** | " % a + " | ".join(str(v) for v in row) + " |")
    lines += [
        "",
        "Book level (majority vote of each book's chunks at %d words): %d of %d books go to the right author."
        % (r["report_n"], r["books_correct"], r["books_total"]),
        "",
    ]
    if r["misattributed"]:
        lines += ["Books that go to the wrong author:", ""]
        for b in r["misattributed"]:
            lines.append(
                "- %s, \"%s\", goes to %s (%s of its chunks)."
                % (b["author"], b["title"], b["predicted"], _pct(b["share"]))
            )
        lines.append("")
    if r["missing"]:
        lines += ["## Books not downloaded", "", "These were skipped, so their authors have less text:", ""]
        lines += ["- %s" % m for m in r["missing"]]
        lines.append("")
    if not r["passed"]:
        lines += [
            "## If this failed",
            "",
            "Check, in order: books that failed to download or were rejected (`style-eval fetch`); "
            "editor prefaces or other non-author text left in a book (`style-eval peek <id>`, then "
            "set `start_at` / `end_at` in the manifest); authors with very little text.",
            "",
        ]
    return "\n".join(lines)
