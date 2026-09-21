#!/usr/bin/env python3
"""Download the archived intermediates for this workflow and verify them.

The bundles are published as a Zenodo record; this script resolves the record,
downloads the tarballs you ask for, unpacks them into the repository, and checks
every file against data/ARTIFACTS.tsv (sha256). Nothing here needs a GPU.

Usage
-----
    python scripts/fetch_artifacts.py results            # ~200 MB: rebuild every table on a CPU
    python scripts/fetch_artifacts.py results s2and      # + 14 GB: re-score name disambiguation
    python scripts/fetch_artifacts.py --verify           # re-check what is already on disk
    python scripts/fetch_artifacts.py --record 1234567 results   # pin a specific record

The APS gene matrices (~190 GB) are deliberately NOT distributed -- they exceed a
Zenodo record and are a deterministic function of the corpus plus the published
checkpoints. Regenerate them with `snakemake all_embeddings`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "ARTIFACTS.tsv"

# The deposition id, which becomes the record id on publish. It resolves only AFTER
# the draft is published (DOI 10.5281/zenodo.22842861); --record overrides it.
ZENODO_RECORD = "22842861"
ZENODO_API = "https://zenodo.org/api/records/{record}"

TIERS = ("results", "s2and")


def load_manifest() -> dict[str, list[tuple[str, int, str]]]:
    if not MANIFEST.exists():
        sys.exit(f"missing {MANIFEST.relative_to(ROOT)} -- it ships with the repository")
    out: dict[str, list[tuple[str, int, str]]] = {}
    for line in MANIFEST.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        tier, path, size, digest = line.split("\t")
        out.setdefault(tier, []).append((path, int(size), digest))
    return out


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify(tier: str, entries: list[tuple[str, int, str]]) -> tuple[int, int, int]:
    ok = bad = missing = 0
    for rel, size, digest in entries:
        p = ROOT / rel
        if not p.exists():
            missing += 1
            continue
        if p.stat().st_size != size or sha256(p) != digest:
            bad += 1
            print(f"  CORRUPT {rel}")
        else:
            ok += 1
    print(f"{tier}: {ok} ok, {bad} corrupt, {missing} missing (of {len(entries)})")
    return ok, bad, missing


def zenodo_files(record: str) -> dict[str, str]:
    url = ZENODO_API.format(record=record)
    with urllib.request.urlopen(url) as resp:                      # noqa: S310 (fixed https host)
        meta = json.load(resp)
    files = {}
    for f in meta.get("files", []):
        name = f.get("key") or f.get("filename")
        link = (f.get("links") or {}).get("self") or (f.get("links") or {}).get("download")
        if name and link:
            files[name] = link
    return files


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("curl"):
        subprocess.run(["curl", "-fL", "--retry", "3", "-o", str(dest), url], check=True)
        return
    with urllib.request.urlopen(url) as resp, dest.open("wb") as fh:  # noqa: S310
        shutil.copyfileobj(resp, fh)


def unpack(tar_path: Path) -> None:
    if tar_path.suffixes[-1:] == [".zst"]:
        if not shutil.which("zstd"):
            sys.exit("this bundle is zstd-compressed; install zstd (apt install zstd / conda install zstd)")
        subprocess.run(["zstd", "-d", "-f", str(tar_path)], check=True)
        tar_path = tar_path.with_suffix("")
    with tarfile.open(tar_path) as tf:
        for member in tf.getmembers():
            target = (ROOT / member.name).resolve()
            if not str(target).startswith(str(ROOT)):             # refuse path traversal
                sys.exit(f"refusing to unpack outside the repository: {member.name}")
        tf.extractall(ROOT)                                        # noqa: S202 (members checked above)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tiers", nargs="*", choices=TIERS, help="which bundles to fetch")
    ap.add_argument("--record", default=ZENODO_RECORD, help="Zenodo record id (numeric)")
    ap.add_argument("--verify", action="store_true", help="only verify what is already unpacked")
    ap.add_argument("--keep-tar", action="store_true", help="do not delete the tarball after unpacking")
    args = ap.parse_args()

    manifest = load_manifest()

    if args.verify:
        bad = 0
        for tier in args.tiers or sorted(manifest):
            _, b, _ = verify(tier, manifest.get(tier, []))
            bad += b
        return 1 if bad else 0

    if not args.tiers:
        ap.error("name at least one tier, or pass --verify")
    if not args.record:
        sys.exit("no Zenodo record id: pass --record <id> (the DOI page shows it), "
                 "or set ZENODO_RECORD at the top of this script once the record is minted")

    available = zenodo_files(args.record)
    staging = ROOT / "dist"
    for tier in args.tiers:
        name = next((n for n in available if n.startswith(f"doc2lora-{tier}.tar")), None)
        if name is None:
            sys.exit(f"record {args.record} has no bundle for tier {tier} (has: {', '.join(sorted(available))})")
        tar_path = staging / name
        print(f"{tier}: downloading {name}")
        download(available[name], tar_path)
        print(f"{tier}: unpacking")
        unpack(tar_path)
        if not args.keep_tar:
            for p in (tar_path, tar_path.with_suffix("")):
                p.unlink(missing_ok=True)
        _, bad, missing = verify(tier, manifest.get(tier, []))
        if bad or missing:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
