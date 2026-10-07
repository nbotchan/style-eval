"""Command line: style-eval fetch | peek | catalog | trustcheck."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .corpus import (
    DEFAULT_MIRROR,
    all_works,
    clean_work,
    fetch_corpus,
    load_manifest,
    mirror_dir,
    raw_path,
)
from .trustcheck import render_report, run_trust_check

DEFAULT_MANIFEST = "corpus/manifest.json"


def cmd_fetch(args) -> int:
    manifest = load_manifest(args.manifest)
    if args.print_rsync:
        # Folder layout is the same on every Gutenberg mirror.
        for _, work in all_works(manifest):
            rel = mirror_dir("", work["id"]).lstrip("/")
            print("rsync -av rsync.ibiblio.org::gutenberg/%s/ mirror/%s/" % (rel, rel))
        return 0
    summary = fetch_corpus(
        manifest, args.data_dir, mirror=args.mirror, local_mirror=args.local_mirror, delay=args.delay
    )
    print(
        "\n%d downloaded, %d already on disk, %d failed, %d rejected."
        % (len(summary["ok"]), len(summary["cached"]), len(summary["failed"]), len(summary["rejected"]))
    )
    return 1 if summary["failed"] or summary["rejected"] else 0


def cmd_peek(args) -> int:
    manifest = load_manifest(args.manifest)
    for _, work in all_works(manifest):
        if work["id"] == args.id:
            path = raw_path(args.data_dir, args.id)
            if not path.exists():
                print("book %d is not downloaded yet" % args.id)
                return 1
            text = clean_work(path.read_text(encoding="utf-8"), work)
            print("--- first %d characters ---\n%s\n" % (args.chars, text[: args.chars]))
            print("--- last %d characters ---\n%s" % (args.chars, text[-args.chars :]))
            return 0
    print("book %d is not in the manifest" % args.id)
    return 1


def cmd_catalog(args) -> int:
    """List English texts by an author from Gutenberg's offline catalog CSV."""
    with open(args.csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if (
                row.get("Language") == "en"
                and row.get("Type") == "Text"
                and args.name.lower() in row.get("Authors", "").lower()
            ):
                print("%s\t%s\t%s" % (row["Text#"], row["Title"].replace("\n", " "), row["Authors"]))
    return 0


def cmd_trustcheck(args) -> int:
    manifest = load_manifest(args.manifest)
    result = run_trust_check(manifest, args.data_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = render_report(result)
    (out / "trustcheck.md").write_text(report, encoding="utf-8")
    (out / "trustcheck.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(report)
    print("\nWrote %s and %s" % (out / "trustcheck.md", out / "trustcheck.json"))
    return 0 if result["passed"] or not args.strict else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="style-eval", description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--data-dir", default="data")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download the books listed in the manifest")
    p.add_argument("--mirror", default=DEFAULT_MIRROR)
    p.add_argument("--local-mirror", help="read from a local rsync copy instead of the network")
    p.add_argument("--delay", type=float, default=2.0, help="seconds between downloads")
    p.add_argument("--print-rsync", action="store_true", help="print rsync commands and exit")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("peek", help="show the start and end of a cleaned book")
    p.add_argument("id", type=int)
    p.add_argument("--chars", type=int, default=500)
    p.set_defaults(func=cmd_peek)

    p = sub.add_parser("catalog", help="search Gutenberg's pg_catalog.csv by author")
    p.add_argument("csv")
    p.add_argument("name")
    p.set_defaults(func=cmd_catalog)

    p = sub.add_parser("trustcheck", help="run the held-out-books attribution check")
    p.add_argument("--out", default="results")
    p.add_argument("--strict", action="store_true", help="exit 1 if the gate fails")
    p.set_defaults(func=cmd_trustcheck)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
