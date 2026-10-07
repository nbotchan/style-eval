"""Burrows' Delta and Cosine Delta, with author-balanced z-scores.

How it works, in four steps:

1. Take the most frequent words (MFW) across the reference books, ignoring words
   that appear in fewer than `cull` of the chunks (this removes character names
   and topic words).
2. Turn each chunk into relative word frequencies, then into z-scores using the
   reference mean and standard deviation. Statistics come from reference text
   only, never from the text being scored.
3. An author's profile is the average z-score vector of their reference chunks.
4. A chunk's distance to an author is Delta (mean absolute z-score difference)
   or Cosine Delta (one minus the cosine of the angle between z-score vectors).
   The nearest author is the attribution.

Every author counts equally in the mean, standard deviation and word ranking,
so an author with a huge corpus (Dickens) does not set the scale for everyone.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

METHODS = ("cosine", "burrows")


def burrows_delta(za: np.ndarray, zb: np.ndarray) -> float:
    """Mean absolute difference of two z-score vectors (Burrows 2002)."""
    return float(np.mean(np.abs(za - zb)))


def cosine_delta(za: np.ndarray, zb: np.ndarray) -> float:
    """One minus the cosine similarity of two z-score vectors (Smith and Aldridge 2011)."""
    denom = float(np.linalg.norm(za) * np.linalg.norm(zb))
    return 1.0 - float(np.dot(za, zb)) / max(denom, 1e-12)


def relative_frequencies(tokens: Sequence[str], index: Dict[str, int]) -> np.ndarray:
    """Counts of the words in `index`, divided by the chunk's total word count."""
    vec = np.zeros(len(index))
    for word, count in Counter(tokens).items():
        i = index.get(word)
        if i is not None:
            vec[i] = count
    return vec / max(len(tokens), 1)


@dataclass
class DeltaModel:
    """A fitted Delta model. Build one with `DeltaModel.fit`."""

    authors: List[str]
    vocab: List[str]  # ranked most-frequent first, after culling
    mu: np.ndarray  # reference mean frequency of each vocab word
    sigma: np.ndarray  # reference standard deviation of each vocab word
    profiles: np.ndarray  # (authors, vocab) mean z-score vector per author

    @classmethod
    def fit(
        cls,
        chunks_by_author: Dict[str, List[List[str]]],
        max_mfw: int = 1000,
        cull: float = 0.5,
    ) -> "DeltaModel":
        authors = sorted(chunks_by_author)
        n_authors = len(authors)

        # Pass 1: author-balanced mean frequency and chunk presence for every word.
        mean_freq: Counter = Counter()
        presence: Counter = Counter()
        for a in authors:
            weight = 1.0 / (n_authors * len(chunks_by_author[a]))
            for chunk in chunks_by_author[a]:
                n = max(len(chunk), 1)
                for word, count in Counter(chunk).items():
                    mean_freq[word] += weight * count / n
                    presence[word] += weight
        candidates = [w for w, p in presence.items() if p >= cull - 1e-9]
        candidates.sort(key=lambda w: (-mean_freq[w], w))
        vocab = candidates[:max_mfw]
        if not vocab:
            raise ValueError("no words survive culling; lower `cull` or add text")
        index = {w: i for i, w in enumerate(vocab)}

        # Pass 2: frequency matrix, author-balanced mean and standard deviation.
        rows, weights, labels = [], [], []
        for ai, a in enumerate(authors):
            weight = 1.0 / (n_authors * len(chunks_by_author[a]))
            for chunk in chunks_by_author[a]:
                rows.append(relative_frequencies(chunk, index))
                weights.append(weight)
                labels.append(ai)
        X = np.vstack(rows)
        W = np.array(weights)
        labels_arr = np.array(labels)
        mu = W @ X
        sigma = np.sqrt(W @ (X - mu) ** 2)
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        Z = (X - mu) / sigma
        profiles = np.vstack([Z[labels_arr == ai].mean(axis=0) for ai in range(n_authors)])
        return cls(authors, vocab, mu, sigma, profiles)

    def transform(self, chunks: Sequence[Sequence[str]]) -> np.ndarray:
        """Z-score vectors (one row per chunk) over the full vocabulary."""
        index = {w: i for i, w in enumerate(self.vocab)}
        X = np.vstack([relative_frequencies(c, index) for c in chunks])
        return (X - self.mu) / self.sigma

    def distances(self, Z: np.ndarray, n_mfw: int, method: str = "cosine") -> np.ndarray:
        """Distance from each chunk to each author using the first `n_mfw` words.

        Returns an array of shape (chunks, authors); smaller means closer.
        """
        if method not in METHODS:
            raise ValueError("method must be one of %s" % (METHODS,))
        n = min(n_mfw, len(self.vocab))
        Zs, P = Z[:, :n], self.profiles[:, :n]
        if method == "burrows":
            return np.stack([np.abs(Zs - p).mean(axis=1) for p in P], axis=1)
        denom = np.linalg.norm(Zs, axis=1)[:, None] * np.linalg.norm(P, axis=1)[None, :]
        return 1.0 - (Zs @ P.T) / np.maximum(denom, 1e-12)

    def attribute(self, Z: np.ndarray, n_mfw: int, method: str = "cosine") -> np.ndarray:
        """Index of the nearest author for each chunk."""
        return np.argmin(self.distances(Z, n_mfw, method), axis=1)
