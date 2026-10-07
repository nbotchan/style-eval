# StylEval

An open-source toolkit for measuring how well language models write in a given style.

The headline idea is **Style Match**, a 0 to 100 score for how far a model's writing
moves from its default voice toward the real thing. Position is measured with
Burrows' Delta and Cosine Delta, standard authorship-attribution methods.

This first milestone is the **trust check**: before any model is scored, the ruler has
to work. Can Delta tell ten authors apart on books it has never seen?

## Run it

```bash
pip install -e .
styleval fetch          # downloads about 100 books from a Project Gutenberg mirror
styleval trustcheck     # writes results/trustcheck.md and results/trustcheck.json
```

Run both from the project folder. Pass `--strict` to `trustcheck` to exit non-zero if
the gate fails. Tests, which need no network: `PYTHONPATH=src python3 -m unittest discover -s tests`.

## What the trust check does

1. Cleans each book (license text, chapter headings, captions removed) and cuts it into
   2,000-word chunks.
2. Splits the books, not the chunks, into three rounds. In each round it fits Delta on two
   thirds of each author's books and holds out the rest. Chunks from one book share
   characters and topic, so splitting by chunk would leak.
3. Assigns every held-out chunk to the nearest of the ten authors.
4. Reports the share assigned correctly, averaged over authors and over 100 to 1,000
   frequent words, with a 95% interval that resamples whole books.

**Gate:** at least 90% (cosine). If it fails, the problem is the setup, not the models.
The report says what to check.

## Design choices worth knowing

- **Every author counts equally** in the word ranking, means and standard deviations, so
  Dickens' 3 million words do not set the scale for everyone.
- **Statistics come from reference text only.** Nothing from the text being scored leaks in.
- **Words must appear in at least half of the chunks** to be used. This drops character
  names and topic words.
- **Wrong IDs are caught.** Every download is checked against the Title and Author lines
  in its own header. A mismatch is reported and the book is skipped.
- **One tokenizer** (lowercase, curly quotes straightened, hyphens split) is used for books
  and, later, for model outputs.

## Getting the books

Gutenberg's website is for human visitors and blocks automated scraping, so `fetch` reads
from a Gutenberg mirror (default `aleph.gutenberg.org`) at one file every 2 seconds. For a
local copy, `styleval fetch --print-rsync` prints rsync commands for exactly the listed
books; point `--local-mirror mirror` at the result. See Gutenberg's
[mirroring guide](https://www.gutenberg.org/help/mirroring.html) and
[robot access policy](https://www.gutenberg.org/policy/robot_access.html).

`styleval catalog pg_catalog.csv Thackeray` lists an author's English texts from Gutenberg's
[offline catalog](https://www.gutenberg.org/ebooks/offline_catalogs.html), for fixing or
extending the book list in `corpus/manifest.json`.

`styleval peek 1342` shows the start and end of a cleaned book, to spot an editor's preface
that cleaning missed. Trim it with `start_at` / `end_at` in the manifest.

## Known weak spots

- The book IDs in the manifest have not been checked against Gutenberg from here; the
  header check exists for that reason. Expect to fix a few.
- Poe's Gutenberg volumes mix poetry, tales and criticism. Swap in Melville if it drags.
- Cleaning is heuristic. Some tables of contents and verse will get through.

## License

MIT. See `LICENSE`. The Gutenberg books are downloaded, not included, and are public domain in the US.

## Layout

```
corpus/manifest.json     books per author, folds, gate
src/styleval/text.py     cleaning, tokenizing, chunking
src/styleval/delta.py    Burrows and Cosine Delta
src/styleval/corpus.py   fetching, verification, chunk building
src/styleval/trustcheck.py  the held-out-books check and its report
tests/                   unit tests, including synthetic-author known-answer tests
```
