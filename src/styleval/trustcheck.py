"""The trust check: can Delta tell the authors apart on books it has never seen?

Before any model is scored, the ruler itself has to work. This test holds out
whole books (never just chunks, because chunks from one book share characters and
topic), fits Delta on the other books, and asks which author each held-out chunk
is closest to. Books are spread over `folds` rounds so every book is held out once.

The gate comes from the spec: at least 90% top-1 in the closed set. We report the
macro average (every author counts equally) over a sweep of MFW sizes, so no single
word-count setting can be cherry-picked.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np

from .corpus import assign_folds, build_chunks
from .delta import DeltaModel

SWEEP = tuple(range(100, 1001, 100))
REPORT_N = 500  # MFW size used for the confusion matrix and per-book results
PRIMARY = "cosine"
METHODS = ("cosine", "burrows")


def _pct(x: float) -> str:
    return "%.1f%%" % (100.0 * x)


def _macro(correct: np.ndarray, truth: np.ndarray, n_authors: int) -> np.ndarray:
    """Per-N macro accuracy: mean over authors of that author's accuracy."""
    return np.mean([correct[:, truth == a].mean(axis=1) for a in range(n_authors)], axis=0)


def _bootstrap_ci(correct, truth, book_of, n_authors, n_boot, rng):
    """95% interval for the headline, resampling whole books within each author."""
    books = []
    for a in range(n_authors):
        ids = sorted({book_of[i] for i in np.where(truth == a)[0]})
        books.append([np.where((truth == a) & (book_of == b))[0] for b in ids])
    values = np.empty(n_boot)
    for i in range(n_boot):
        per_author = []
        for author_books in books:
            pick = rng.integers(0, len(author_books), size=len(author_books))
            idx = np.concatenate([author_books[j] for j in pick])
            per_author.append(correct[:, idx].mean())
        values[i] = np.mean(per_author)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def run_trust_check(manifest: dict, data_dir, sweep=SWEEP, n_boot: int = 500, seed: int = 0) -> dict:
    chunks, missing = build_chunks(manifest, data_dir)
    authors = sorted({c.author for c in chunks})
    if len(authors) < 2:
        raise ValueError("need text from at least two authors; run `styleval fetch` first")
    books: Dict[str, List[int]] = {a: sorted({c.work_id for c in chunks if c.author == a}) for a in authors}
    thin = [a for a, b in books.items() if len(b) < 2]
    if thin:
        raise ValueError("need at least 2 books per author; too few for: %s" % ", ".join(thin))
    k = min(manifest.get("folds", 3), min(len(b) for b in books.values()))

    fold_of: Dict[int, int] = {}
    for a in authors:
        fold_of.update(assign_folds(books[a], k, "%s-%s" % (manifest.get("seed", ""), a)))

    cull = manifest.get("cull", 0.5)
    n_authors = len(authors)
    truth = np.array([authors.index(c.author) for c in chunks])
    book_of = np.array([c.work_id for c in chunks])
    fold_ids = np.array([fold_of[c.work_id] for c in chunks])
    methods = METHODS

    # Fit one model per round. Culling can leave fewer words than the largest sweep size,
    # so the sweep is trimmed to what every round actually has (no duplicated columns).
    models = []
    for f in range(k):
        ref: Dict[str, List[List[str]]] = {a: [] for a in authors}
        for c, fid in zip(chunks, fold_ids):
            if fid != f:
                ref[c.author].append(c.tokens)
        models.append(DeltaModel.fit(ref, max_mfw=max(sweep), cull=cull))
    vocab_sizes = [len(model.vocab) for model in models]
    sweep = tuple(n for n in sweep if n <= min(vocab_sizes)) or (min(vocab_sizes),)

    pred = {m: np.zeros((len(sweep), len(chunks)), dtype=int) for m in methods}
    for f, model in enumerate(models):
        test_idx = np.where(fold_ids == f)[0]
        if len(test_idx) == 0:
            continue
        Z = model.transform([chunks[i].tokens for i in test_idx])
        for m in methods:
            for ni, n in enumerate(sweep):
                pred[m][ni, test_idx] = model.attribute(Z, n, m)

    rng = np.random.default_rng(seed)
    report_i = int(np.argmin([abs(n - REPORT_N) for n in sweep]))
    titles = {c.work_id: c.title for c in chunks}
    result: dict = {
        "authors": authors,
        "n_chunks": len(chunks),
        "chunk_size": manifest.get("chunk_size", 2000),
        "books_per_author": {a: len(books[a]) for a in authors},
        "chunks_per_author": {a: int((truth == i).sum()) for i, a in enumerate(authors)},
        "folds": k,
        "sweep": list(sweep),
        "report_n": sweep[report_i],
        "vocab_sizes": vocab_sizes,
        "missing": missing,
        "gate": manifest.get("gate", 0.9),
        "primary": PRIMARY,
        "methods": {},
    }

    for m in methods:
        correct = pred[m] == truth[None, :]
        macro = _macro(correct, truth, n_authors)
        headline = float(macro.mean())
        lo, hi = _bootstrap_ci(correct, truth, book_of, n_authors, n_boot, rng)
        confusion = np.zeros((n_authors, n_authors), dtype=int)
        for t, p in zip(truth, pred[m][report_i]):
            confusion[t, p] += 1
        pairs = []
        for a, b in manifest.get("pairs", []):
            if a in authors and b in authors:
                ia, ib = authors.index(a), authors.index(b)
                ab = np.mean(pred[m][:, truth == ia] == ib)
                ba = np.mean(pred[m][:, truth == ib] == ia)
                pairs.append({"a": a, "b": b, "a_as_b": float(ab), "b_as_a": float(ba)})
        # Book-level: majority vote of a held-out book's chunks at REPORT_N.
        book_results, misattributed = [], []
        for wid in sorted(set(book_of.tolist())):
            sel = np.where(book_of == wid)[0]
            votes = Counter(pred[m][report_i][sel].tolist())
            winner, count = votes.most_common(1)[0]
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
        result["methods"][m] = {
            "macro_by_n": [float(x) for x in macro],
            "micro_by_n": [float(x) for x in correct.mean(axis=1)],
            "headline": headline,
            "ci": [lo, hi],
            "per_author": {a: float(correct[:, truth == i].mean()) for i, a in enumerate(authors)},
            "confusion": confusion.tolist(),
            "pairs": pairs,
            "books_correct": int(sum(book_results)),
            "books_total": len(book_results),
            "misattributed": misattributed,
        }
    result["passed"] = result["methods"][PRIMARY]["headline"] >= result["gate"]
    return result


def render_report(r: dict) -> str:
    """Turn a trust-check result into a short Markdown report."""
    p = r["methods"][r["primary"]]
    verdict = "PASS" if r["passed"] else "FAIL"
    lines = [
        "# StylEval trust check",
        "",
        "**%s.** Cosine Delta names the right author for %s of held-out text "
        "(95%% interval %s to %s; gate %s)."
        % (verdict, _pct(p["headline"]), _pct(p["ci"][0]), _pct(p["ci"][1]), _pct(r["gate"])),
        "",
        "The test: whole books are held out, Delta is fitted on the rest, and each "
        "%d-word chunk of a held-out book is assigned to the nearest of %d authors. "
        "The score is the share assigned correctly, averaged over authors (so a big "
        "author cannot carry it) and over %d to %d frequent words (so no single "
        "setting can be picked). If real text by an author is not attributed to them, "
        "the ruler is broken, not the model. Frequent-word counts above the number of "
        "words that survive culling are left out of the sweep."
        % (r["chunk_size"], len(r["authors"]), r["sweep"][0], r["sweep"][-1]),
        "",
        "## Setup",
        "",
        "- %d authors, %d chunks, %d rounds of held-out books."
        % (len(r["authors"]), r["n_chunks"], r["folds"]),
        "- Words kept after culling, per round: %s." % ", ".join(str(v) for v in r["vocab_sizes"]),
        "",
        "## Headline by method",
        "",
        "| Method | Right, averaged over authors | 95% interval | Right, all chunks |",
        "| --- | --- | --- | --- |",
    ]
    for m, d in r["methods"].items():
        micro = float(np.mean(d["micro_by_n"]))
        lines.append(
            "| %s | %s | %s to %s | %s |"
            % (m, _pct(d["headline"]), _pct(d["ci"][0]), _pct(d["ci"][1]), _pct(micro))
        )
    lines += [
        "",
        "Intervals resample whole books, since chunks from one book are not independent.",
        "",
        "## Accuracy by number of frequent words (cosine)",
        "",
        "| Words | " + " | ".join(str(n) for n in r["sweep"]) + " |",
        "| --- |" + " --- |" * len(r["sweep"]),
        "| Right | " + " | ".join(_pct(x) for x in p["macro_by_n"]) + " |",
        "",
        "## Per author (cosine, averaged over the sweep)",
        "",
        "| Author | Books | Chunks | Right |",
        "| --- | --- | --- | --- |",
    ]
    for a in r["authors"]:
        lines.append(
            "| %s | %d | %d | %s |"
            % (a, r["books_per_author"][a], r["chunks_per_author"][a], _pct(p["per_author"][a]))
        )
    lines += ["", "## Where it goes wrong", ""]
    if p["pairs"]:
        lines += ["Close pairs, averaged over the sweep (how often one is mistaken for the other):", ""]
        for pr in p["pairs"]:
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
    for a, row in zip(r["authors"], p["confusion"]):
        lines.append("| **%s** | " % a + " | ".join(str(v) for v in row) + " |")
    lines += [
        "",
        "Book level (majority vote of each book's chunks at %d words): %d of %d books go to the right author."
        % (r["report_n"], p["books_correct"], p["books_total"]),
        "",
    ]
    if p["misattributed"]:
        lines += ["Books that go to the wrong author:", ""]
        for b in p["misattributed"]:
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
            "Check, in order: books that failed to download or were rejected (`styleval fetch`); "
            "editor prefaces or other non-author text left in a book (`styleval peek <id>`, then "
            "set `start_at` / `end_at` in the manifest); authors with very little text.",
            "",
        ]
    return "\n".join(lines)
