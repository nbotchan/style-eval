# Style Eval

An open-source toolkit for measuring how well large language models (LLMs) can emulate 
people's unique, individual styles of writing, called an *idiolect*. Just like a 
dialect is shared by a group of people, usually based on geographic region, culture, 
or social class, an idiolect belongs to a single person. 

Idiolects are linguistic fingerprints—no two people share the exact same vocabulary, 
grammar, catchphrases, and quirks of language. They constantly evolve as people age, 
learn new words and phrases, and experience formative events in life. 

The key metric in Style Eval is **Style Match**, a 0 to 100 score for how far a model's
writing moves from its default voice toward a specific idiolect. Under the hood, this
is measured with Burrows' Delta, a standard authorship-attribution method, as implemented
by [faststylometry](https://pypi.org/project/faststylometry/).

## Set up

Style Eval needs Python 3.11 or newer and runs from a project virtualenv, so it never
touches your system or conda Python. Build it once with the setup script:

```bash
scripts/setup_env.sh
```

The script finds a suitable Python, creates `.venv/`, and installs the package in editable
mode with the test dependencies. Use `scripts/setup_env.sh --fresh` to rebuild from scratch,
or `PYTHON=/path/to/python3.12 scripts/setup_env.sh` to choose the interpreter.

Then activate the environment in each new shell:

```bash
source .venv/bin/activate
```

Running a bare `pip install -e .` can pick up the wrong `pip` (an old conda or system one)
and fail with `setup.py not found`. Activate `.venv` first, or use the script.

## Run it

With `.venv` activated and from the project folder:

```bash
style-eval fetch          # downloads about 100 books from a Project Gutenberg mirror
style-eval trustcheck     # writes results/trustcheck.md and results/trustcheck.json
```

For a quick look, `style-eval fetch --sample` downloads just the first book for each author
(10 books instead of about 100). That is enough to try the tools, but too little for a
meaningful trust check, which needs held-out books.

Pass `--strict` to `trustcheck` to exit non-zero if the gate fails. Tests, which need no
network: `python -m pytest`.

## What the trust check does

1. Cleans each book (license text, chapter headings, captions removed) and cuts it into
   2,000-word chunks.
2. Splits the books, not the chunks, into three rounds. In each round it fits Delta on two
   thirds of each author's books and holds out the rest. Chunks from one book share
   characters and topic, so splitting by chunk would leak.
3. Assigns every held-out chunk to the nearest of the ten authors, using faststylometry's
   `calculate_burrows_delta`.
4. Reports the share assigned correctly, averaged over authors and over 100, 300 and 500
   frequent words.

**Gate:** at least 90%. If it fails, the problem is the setup, not the models.
The report says what to check.

**Current status: the gate fails.** On the full 92-book corpus, faststylometry's Burrows'
Delta scores about 75% (71% to 77% across 100 to 500 words). Austen and Poe are near 93%;
Dickens (59%) and James (65%) are weakest. Nothing has been tuned to reach the gate. Likely
causes: faststylometry does not drop character names and topic words, removes pronouns, and
offers only Burrows' Delta. Until this passes, do not trust Style Match scores.

## Design choices worth knowing

- **Delta is faststylometry's.** Word ranking, z-scores, pronoun removal and tokenizing
  are all its defaults. Style Eval only fetches, cleans and chunks the books and splits
  them into rounds.
- **Statistics come from reference text only.** Nothing from the text being scored leaks in.
- **Wrong IDs are caught.** Every download is checked against the Title and Author lines
  in its own header. A mismatch is reported and the book is skipped.
- **One tokenizer** (faststylometry's, which drops pronouns) is used for books and, later,
  for model outputs.

## Getting the books

Gutenberg's website is for human visitors and blocks automated scraping, so `fetch` reads
from a Gutenberg mirror (default `aleph.pglaf.org`) at one file every 2 seconds. For a
local copy, `style-eval fetch --print-rsync` prints rsync commands for exactly the listed
books; point `--local-mirror mirror` at the result. See Gutenberg's
[mirroring guide](https://www.gutenberg.org/help/mirroring.html) and
[robot access policy](https://www.gutenberg.org/policy/robot_access.html).

`style-eval catalog pg_catalog.csv Thackeray` lists an author's English texts from Gutenberg's
[offline catalog](https://www.gutenberg.org/ebooks/offline_catalogs.html), for fixing or
extending the book list in `corpus/manifest.json`.

`style-eval peek 1342` shows the start and end of a cleaned book, to spot an editor's preface
that cleaning missed. Trim it with `start_at` / `end_at` in the manifest.

## Known weak spots

- The book IDs in the manifest have not been checked against Gutenberg from here; the
  header check exists for that reason. Expect to fix a few.
- Poe's Gutenberg volumes mix poetry, tales and criticism. Swap in Melville if it drags.
- The trust check fails its gate with faststylometry's defaults (see above).
- Cleaning is heuristic. Some tables of contents and verse will get through.

## License

MIT. See `LICENSE`. The Gutenberg books are downloaded, not included, and are public domain in the US.

## Layout

```
corpus/manifest.json     books per author, folds, gate
src/style_eval/text.py     cleaning and chunking
src/style_eval/corpus.py   fetching, verification, chunk building
src/style_eval/trustcheck.py  the held-out-books check and its report
tests/                   unit tests, including synthetic-author known-answer tests
```
