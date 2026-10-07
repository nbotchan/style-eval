"""Text cleaning, tokenizing and chunking.

Everything here is deterministic and uses only the standard library. The same
tokenizer is used for reference books and (later) for model outputs, so the two
are always measured the same way.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

# --- Project Gutenberg boilerplate ------------------------------------------------

_START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S
)
_END_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.I)


def split_gutenberg(raw: str) -> Tuple[str, str]:
    """Split a Gutenberg file into (header, body).

    The header holds the Title/Author lines; the body is the book itself with the
    license text removed. Raises ValueError if the standard START marker is missing.
    """
    start = _START_RE.search(raw)
    if not start:
        raise ValueError("no Project Gutenberg START marker found")
    header = raw[: start.start()]
    rest = raw[start.end():]
    end = _END_RE.search(rest)
    body = rest[: end.start()] if end else rest
    return header, body


def header_field(header: str, field: str) -> Optional[str]:
    """Read a 'Title:' or 'Author:' style line from the Gutenberg header."""
    match = re.search(r"^%s:\s*(.+)$" % re.escape(field), header, re.M | re.I)
    return match.group(1).strip() if match else None


# --- Typography -------------------------------------------------------------------

_TYPO = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
        "\ufeff": "",
    }
)


def normalize_typography(text: str) -> str:
    """Curly quotes to straight, en/em dashes to hyphens, no-break spaces to spaces."""
    return text.translate(_TYPO)


# --- Cleaning the book body --------------------------------------------------------

_BRACKET_RE = re.compile(
    r"\[(?:illustration|footnote|transcriber|note|sidenote|image)[^\]]*\]", re.I | re.S
)
_BOILER_RE = re.compile(
    r"^(?:produced by|transcribed from|e-?text prepared|updated editions|"
    r"this file was produced|special thanks|\[?transcriber)",
    re.I,
)
_HEADING_RE = re.compile(
    r"^\s*(?i:chapter|book|part|volume|vol\.|letter|canto|act|scene|section)"
    r"\s+(?:[IVXLCDM]+|\d+)\b.*$"
)
_NUMERAL_RE = re.compile(r"^\s*(?:[IVXLCDM]+|\d+)\.?\s*$")


def _is_shouting(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    return bool(letters) and len(line.strip()) <= 80 and all(c.isupper() for c in letters)


def _is_furniture(line: str) -> bool:
    """Chapter headings, bare numerals, ALL-CAPS titles and 'Contents' lines."""
    return (
        bool(_HEADING_RE.match(line))
        or bool(_NUMERAL_RE.match(line))
        or _is_shouting(line)
        or line.strip().lower() == "contents"
    )


def clean_body(body: str) -> str:
    """Remove book furniture so that only running prose is left.

    Drops illustration and footnote brackets, transcriber credits, chapter
    headings, bare numerals, ALL-CAPS title lines and 'Contents'. This is a
    heuristic: use `styleval peek <id>` to spot anything it misses (for example
    an editor's preface) and trim it with `start_at` / `end_at` in the manifest.
    """
    text = normalize_typography(body.replace("\r\n", "\n"))
    text = _BRACKET_RE.sub(" ", text)
    kept = []
    for para in re.split(r"\n\s*\n", text):
        if _BOILER_RE.match(para.strip()):
            continue
        lines = [ln.strip() for ln in para.split("\n") if not _is_furniture(ln)]
        lines = [ln for ln in lines if ln]
        if lines:
            kept.append(" ".join(lines))
    return "\n\n".join(kept)


def apply_window(text: str, start_at: Optional[str] = None, end_at: Optional[str] = None) -> str:
    """Keep only the text from the first `start_at` up to the first `end_at`.

    Used to cut editor prefaces, appendices and the like. Raises ValueError if a
    marker is given but not found, so a typo cannot silently keep the wrong text.
    """
    if start_at:
        i = text.find(start_at)
        if i < 0:
            raise ValueError("start_at marker not found: %r" % start_at)
        text = text[i:]
    if end_at:
        j = text.find(end_at)
        if j < 0:
            raise ValueError("end_at marker not found: %r" % end_at)
        text = text[:j]
    return text


# --- Tokens and chunks -------------------------------------------------------------

_WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens. Keeps internal apostrophes ("don't"), splits on hyphens."""
    return _WORD_RE.findall(normalize_typography(text).lower())


def chunk_tokens(tokens: List[str], size: int = 2000, min_size: int = 1000) -> List[List[str]]:
    """Cut a token list into consecutive chunks of `size` words.

    The last chunk is dropped if it has fewer than `min_size` words, because
    short chunks make Delta noisy. Delta works on word counts, so chunk edges do
    not need to fall on sentence boundaries.
    """
    chunks = [tokens[i : i + size] for i in range(0, len(tokens), size)]
    if chunks and len(chunks[-1]) < min_size:
        chunks.pop()
    return chunks
