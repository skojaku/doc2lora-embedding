#!/usr/bin/env python3
"""Build the anonymized code supplement for a double-blind submission, and prove it.

ICLR is double blind and the rule is strict: "Any paper where author identity is
revealed in either the main text or the supplementary material will be desk
rejected." The guide's own remedies are an anonymized zip in the supplementary
material, an anonymous repository link, or an anonymous link posted in the
discussion forum. This script builds the zip, which depends on no external service.

The verifier is the point of the script, not the copying. An earlier version of it
scanned only files it could decode as text, so three leaks walked straight through:

  * figs/*.pdf            PDF DocInfo carried  Author: skojaku-admin
  * figs/*.graffle        an OmniGraffle bundle is a zip; its data.plist carried
                          Creator/Modifier with the author's full name
  * LICENSE               a copyright holder who is not an author of this paper but
                          is a frequent co-author, which leaks the lab lineage

So the scanner now reads every file as BYTES, tries several text encodings,
inflates compressed PDF streams, and recurses into zip containers. Binary formats
are where identity hides, which is exactly where the old check was blind.

    python scripts/make_anon_supplement.py                 # -> dist/anonymous-code-supplement.zip
    python scripts/make_anon_supplement.py --check-only    # report what would leak
    python scripts/make_anon_supplement.py --audit-raw     # scan the repo as-is; proves
                                                           # the scanner finds the leaks
                                                           # the drop/rewrite rules remove
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Excluded from the supplement entirely (prefix match).
DROP_PREFIXES = (
    "paper/",                       # the PDF is the submission; main.tex has the author block
    "CITATION.cff",                 # names, email, affiliation
    "scripts/zenodo_upload.py",     # embeds the creator list
    # This script itself: its rewrite table has to spell out the names it removes,
    # so shipping it would undo the work. The verifier caught exactly that.
    "scripts/make_anon_supplement.py",
    ".github/",
    # Binary figure assets. Their metadata names the author (PDF DocInfo, OmniGraffle
    # data.plist) and nothing in the supplement consumes them: paper/ is excluded, so
    # the manuscript is not built here, and the reviewer already has the figures in the
    # submitted PDF. Dropping them removes a whole leak class instead of scrubbing
    # proprietary metadata and hoping.
    "figs/doc2lora-figs.pdf",
    "figs/doc2lora-figs.graffle",
    "figs/pacs-clustering.pdf",
)

# Every name that must not appear, in any file, in any encoding. Short or
# common-substring names get a word boundary so the check stays useful rather than
# noisy; distinctive ones match anywhere.
FORBIDDEN = (
    r"kojaku",
    r"skojaku",
    r"binghamton",
    r"mansuri",
    r"cmansur",
    r"zachariah",
    r"yong[- ]?yeol",
    r"\bahn\b",
    r"doc2lora-repro",
    r"doc2lora-embedding",
)

# Literal rewrites applied to every staged TEXT file, in order.
REWRITES = (
    ('{ name = "Sadamori Kojaku", email = "skojaku@binghamton.edu" },',
     '{ name = "Anonymous" },'),
    ('#   pip install "doc2lora @ git+https://github.com/skojaku/doc2lora-embedding.git#subdirectory=libs/doc2lora"',
     '#   pip install "doc2lora @ <repository URL, withheld for double-blind review>#subdirectory=libs/doc2lora"'),
    ('Homepage = "https://github.com/skojaku/doc2lora-embedding"',
     'Homepage = "<withheld for double-blind review>"'),
    ('Repository = "https://github.com/skojaku/doc2lora-embedding"',
     'Repository = "<withheld for double-blind review>"'),
    ("https://github.com/skojaku/doc2lora-repro",
     "<repository URL, withheld for double-blind review>"),
    ("https://github.com/skojaku/doc2lora-embedding",
     "<development repository, withheld for double-blind review>"),
    ("github.com/skojaku/doc2lora-repro",
     "<repository, withheld for double-blind review>"),
    ("`skojaku/doc2lora-repro`", "this workflow"),
    ("skojaku/doc2lora-repro", "this workflow"),
    ("skojaku/doc2lora-embedding", "the development repository"),
    ('ZENODO_RECORD = "22842861"',
     "ZENODO_RECORD = None          # archive DOI withheld for double-blind review"),
    ("(DOI 10.5281/zenodo.22842861)", "(DOI withheld for double-blind review)"),
    ("/home/skojaku/projects/doc2lora-embedding", "<repository root>"),
    # The MIT text is kept; only the holder is withheld for review.
    ("Copyright (c) 2026 Sadamori Kojaku",
     "Copyright (c) 2026 Anonymous authors (holder withheld for double-blind review)"),
)

BANNER = """ANONYMIZED SUPPLEMENT
=====================

This is the reproduction workflow for the submission, with author-identifying
information removed for double-blind review: repository URLs, package author
fields, the copyright holder, the archive DOI, the manuscript sources (you already
have the PDF), and the binary figure assets (their embedded metadata names the
author, and nothing here consumes them).

Start with REPRODUCE.md, which maps every figure, table and quoted number in the
paper to the rule that produces it, and README.md for the three entry points.
`snakemake -n paper_assets` prints the whole 157-job DAG without running anything.

The archived intermediates that let the tables rebuild on a CPU in minutes are
deposited in a public archive; its DOI is withheld here because the record names
its depositor. It will be cited in the camera-ready version.
"""

PAT = re.compile("|".join(FORBIDDEN), re.I)
ENCODINGS = ("utf-8", "utf-16-le", "utf-16-be", "latin-1")


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    return [l for l in out.stdout.splitlines() if l]


def is_text(path: Path) -> bool:
    try:
        path.read_text()
        return True
    except (UnicodeDecodeError, IsADirectoryError):
        return False


def inflated_streams(data: bytes, limit: int = 400):
    """Yield the inflated contents of zlib streams (PDF FlateDecode objects, XMP)."""
    for i, m in enumerate(re.finditer(rb"stream\r?\n", data)):
        if i >= limit:
            return
        start = m.end()
        end = data.find(b"endstream", start)
        blob = data[start:end if end != -1 else min(start + (1 << 20), len(data))]
        try:
            yield zlib.decompressobj().decompress(blob)
        except zlib.error:
            continue


def hits_in_bytes(data: bytes, where: str, depth: int = 0) -> list[tuple[str, str]]:
    """Every forbidden-name hit in one blob: raw in several encodings, inflated
    streams, and zip members (a .graffle/.docx is a zip)."""
    out: list[tuple[str, str]] = []
    for enc in ENCODINGS:
        for m in PAT.finditer(data.decode(enc, "ignore")):
            out.append((where if enc == "utf-8" else f"{where} [{enc}]", m.group(0)))
            break                                     # one hit per encoding is enough
    for chunk in inflated_streams(data):
        for m in PAT.finditer(chunk.decode("utf-8", "ignore")):
            out.append((f"{where} [compressed stream]", m.group(0)))
            break
    if depth < 2 and data[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(__import__("io").BytesIO(data)) as z:
                for name in z.namelist()[:200]:
                    try:
                        out += hits_in_bytes(z.read(name), f"{where}!{name}", depth + 1)
                    except (KeyError, RuntimeError, zlib.error):
                        continue
        except zipfile.BadZipFile:
            pass
    # de-duplicate, keep order
    seen, uniq = set(), []
    for h in out:
        if h not in seen:
            seen.add(h); uniq.append(h)
    return uniq


def scan_tree(tree: Path) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for p in sorted(tree.rglob("*")):
        if p.is_file():
            found += hits_in_bytes(p.read_bytes(), str(p.relative_to(tree)))
    return found


def stage(dest: Path, apply_rules: bool = True) -> list[str]:
    kept = []
    for rel in tracked_files():
        if apply_rules and any(rel == d or rel.startswith(d) for d in DROP_PREFIXES):
            continue
        src, dst = ROOT / rel, dest / rel
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if apply_rules and is_text(dst):
            t = dst.read_text()
            for old, new in REWRITES:
                t = t.replace(old, new)
            dst.write_text(t)
        kept.append(rel)
    if apply_rules:
        (dest / "READ_ME_FIRST.txt").write_text(BANNER)
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "dist" / "anonymous-code-supplement.zip")
    ap.add_argument("--check-only", action="store_true", help="report leaks, write nothing")
    ap.add_argument("--audit-raw", action="store_true",
                    help="scan the tracked tree with NO drops or rewrites: shows what the "
                         "rules are removing, and proves the scanner sees it")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        tree = Path(td) / "supplement"
        tree.mkdir()
        kept = stage(tree, apply_rules=not args.audit_raw)
        found = scan_tree(tree)

        if args.audit_raw:
            print(f"raw audit of {len(kept)} tracked files: {len(found)} leak(s)")
            for where, tok in found:
                print(f"  {where}: {tok}")
            return 0

        if found:
            print(f"LEAK: {len(found)} hit(s) still name the authors:", file=sys.stderr)
            for where, tok in found[:40]:
                print(f"  {where}: {tok}", file=sys.stderr)
            print("\nAdd a rewrite or a drop rule for each, then re-run.", file=sys.stderr)
            return 1

        print(f"clean: {len(kept)} files scanned as bytes (text encodings, compressed "
              f"streams, zip members); no hit for any of {len(FORBIDDEN)} name patterns")
        if args.check_only:
            return 0

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for p in sorted(tree.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(tree))
        print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB, {len(kept) + 1} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
