"""Reference corpus: the manifest, fetching from a Gutenberg mirror, and chunking.

Books are listed in `corpus/manifest.json` by Gutenberg ID. Every downloaded file
is checked against the Title and Author lines in its own header, so a wrong ID
is reported and skipped instead of silently polluting an author's corpus.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .text import (
    apply_window,
    chunk_tokens,
    clean_body,
    header_field,
    split_gutenberg,
    tokenize,
)

USER_AGENT = "styleval/0.0.1 (open-source stylometry research; polite, rate-limited)"
DEFAULT_MIRROR = "https://aleph.gutenberg.org"
SUFFIXES = ("-0.txt", ".txt", "-8.txt")  # UTF-8, ASCII, Latin-1 editions


def load_manifest(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def raw_path(data_dir, work_id: int) -> Path:
    return Path(data_dir) / "raw" / ("%d.txt" % work_id)


def mirror_dir(mirror: str, work_id: int) -> str:
    """Gutenberg's folder layout: ID 1342 lives at <mirror>/1/3/4/1342/."""
    s = str(work_id)
    if len(s) < 2:
        raise ValueError("work IDs below 10 use a different layout and are not supported")
    return "%s/%s/%s" % (mirror.rstrip("/"), "/".join(s[:-1]), s)


def candidate_urls(mirror: str, work_id: int) -> List[str]:
    base = mirror_dir(mirror, work_id)
    return ["%s/%d%s" % (base, work_id, suffix) for suffix in SUFFIXES]


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def verify_header(header: str, author_key: str, expect_title: str) -> Optional[str]:
    """Return None if the header matches what the manifest expects, else a reason."""
    author = (header_field(header, "Author") or "").lower()
    title = (header_field(header, "Title") or "").lower()
    if author_key.lower() not in author:
        return "author is %r, expected %r" % (author, author_key)
    if expect_title.lower() not in title:
        return "title is %r, expected it to contain %r" % (title, expect_title)
    return None


def all_works(manifest: dict):
    """Yield (author_key, work_dict) for every book in the manifest."""
    for author, info in manifest["authors"].items():
        for work in info["works"]:
            yield author, work


def fetch_corpus(
    manifest: dict,
    data_dir,
    mirror: str = DEFAULT_MIRROR,
    local_mirror: Optional[str] = None,
    delay: float = 2.0,
    log=print,
) -> Dict[str, List[str]]:
    """Download every book in the manifest that is not already on disk.

    `local_mirror` points at a folder laid out like Gutenberg's (for example one
    made with rsync) and is read instead of the network. Returns a summary with
    lists of 'ok', 'cached', 'failed' (cannot download) and 'rejected' (wrong book).
    """
    summary: Dict[str, List[str]] = {"ok": [], "cached": [], "failed": [], "rejected": []}
    for author, work in all_works(manifest):
        wid = work["id"]
        label = "%s #%d (%s)" % (author, wid, work["expect"])
        target = raw_path(data_dir, wid)
        if target.exists():
            summary["cached"].append(label)
            continue
        text = None
        for url in candidate_urls(local_mirror or mirror, wid):
            try:
                if local_mirror:
                    p = Path(url)
                    if not p.exists():
                        continue
                    text = _decode(p.read_bytes())
                else:
                    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        text = _decode(resp.read())
                    time.sleep(delay)
                break
            except urllib.error.HTTPError as err:
                if err.code != 404:
                    log("  %s: HTTP %d from %s" % (label, err.code, url))
            except (urllib.error.URLError, OSError) as err:
                log("  %s: %s" % (label, err))
                break
        if text is None:
            summary["failed"].append(label)
            log("FAILED   %s" % label)
            continue
        try:
            header, _ = split_gutenberg(text)
        except ValueError as err:
            summary["rejected"].append("%s: %s" % (label, err))
            log("REJECTED %s: %s" % (label, err))
            continue
        problem = verify_header(header, author, work["expect"])
        if problem:
            summary["rejected"].append("%s: %s" % (label, problem))
            log("REJECTED %s: %s" % (label, problem))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        summary["ok"].append(label)
        log("ok       %s" % label)
    return summary


def clean_work(raw_text: str, work: dict) -> str:
    """Raw Gutenberg file to clean running prose, honoring start_at / end_at."""
    _, body = split_gutenberg(raw_text)
    text = clean_body(body)
    return apply_window(text, work.get("start_at"), work.get("end_at"))


@dataclass
class Chunk:
    author: str
    work_id: int
    title: str
    tokens: List[str]


def build_chunks(manifest: dict, data_dir) -> Tuple[List[Chunk], List[str]]:
    """Chunk every downloaded book. Returns (chunks, labels of books not on disk)."""
    size = manifest.get("chunk_size", 2000)
    min_size = manifest.get("min_chunk", 1000)
    chunks: List[Chunk] = []
    missing: List[str] = []
    for author, work in all_works(manifest):
        path = raw_path(data_dir, work["id"])
        if not path.exists():
            missing.append("%s #%d (%s)" % (author, work["id"], work["expect"]))
            continue
        tokens = tokenize(clean_work(path.read_text(encoding="utf-8"), work))
        for piece in chunk_tokens(tokens, size, min_size):
            chunks.append(Chunk(author, work["id"], work["expect"], piece))
    return chunks, missing


def assign_folds(work_ids: List[int], k: int, seed: str) -> Dict[int, int]:
    """Spread an author's works over k folds, deterministically."""
    ids = sorted(work_ids)
    random.Random(seed).shuffle(ids)
    return {wid: i % k for i, wid in enumerate(ids)}
