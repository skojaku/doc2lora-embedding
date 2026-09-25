#!/usr/bin/env python3
"""Build the Zenodo artifact bundles from a completed run of this workflow.

Reproducing every number from raw corpora needs ~200 GB of intermediates and
several GPU-days. The bundles let someone skip parts of that:

  results  (~200 MB)  every scored / aggregated artifact downstream of the
                      embeddings: bootstrap pools, S2AND signature tables,
                      adapter weights, label-eval JSON, the PACS node set.
                      With this bundle alone, `snakemake paper_assets` rebuilds
                      every table in the manuscript on a CPU.
  s2and    (~14 GB)   the per-benchmark gene / text embeddings for the five
                      author-disambiguation datasets, so the name-disambiguation
                      rows can be re-scored from the vectors up.
  aps      (~190 GB)  the 644k-paper APS gene matrices. Too large for a Zenodo
                      record; listed here so the manifest is honest about what
                      is NOT distributed. Regenerate with `snakemake
                      all_embeddings` (3 encoders x 644k papers).

Usage
-----
    python scripts/make_artifact_bundle.py --list                 # what would go in
    python scripts/make_artifact_bundle.py results --out dist/    # stage + hash + tar
    python scripts/make_artifact_bundle.py results s2and --out dist/

Each bundle writes <out>/<tier>.tar.zst (or .tar.gz without zstd), plus a
manifest row per file in data/ARTIFACTS.tsv: tier, path, bytes, sha256. The
manifest is what scripts/fetch_artifacts.py verifies against.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "ARTIFACTS.tsv"
# Where the completed run lives. Defaults to this repository; point --root at the
# development checkout when the artifacts were produced there.
SRC = ROOT

# Glob patterns per tier, relative to the repository root. Paths are the CURRENT
# layout (code in workflow/, results under data/); fetch_artifacts.LEGACY_LAYOUT
# reads the older bundles onto it. Order matters only for
# readability; a file is assigned to the first tier that claims it.
TIERS: dict[str, list[str]] = {
    "results": [
        # bootstrap CIs behind the two similarity tables
        "data/uncertainty/pools/*.parquet",
        "data/uncertainty/uncertainty_summary.csv",
        "data/uncertainty/bootstrap_replicates.parquet",
        # per-task score tables
        "data/kron/results_*.csv",
        "data/s2and/results_*.csv",
        "data/s2and/proc/*/sig_table.parquet",
        # invertible adapters (small: ~10^5-10^7 params)
        "data/kron/adapter_*.pt",
        "data/general_adapter/adapter_general_*.pt",
        "data/general_adapter/leakage_overlap.json",
        "data/general_adapter/triplets_1x.parquet",
        "data/general_adapter/triplets.parquet",
        # cluster-labelling chain
        "data/pacs/results/*.parquet",
        "data/pacs/results/tree.json",
        "data/labels/*.json",
        "data/labels/*.csv",
        "data/labels/*.tex",
        "data/labels/qwen_fullrank_means.npz",
        # appendix controls
        "data/groupc/psens/*.json",
        "data/groupc/psens/*.parquet",
        "data/groupc/incoherent/*.json",
        "data/groupc/incoherent/*.parquet",
        "data/groupc/fidelity/sample.parquet",
        # benchmark pair lists + the pooling rho
        "data/collab_scores_aps_*.parquet",
        "data/aps/pooling_spearman.csv",
        # the manuscript tables themselves (so a reader can diff their rebuild)
        "results/figs/similarity_benchmarks.tex",
        "results/figs/encoder_matrix.tex",
        "results/figs/prompt_sensitivity.tex",
        "results/figs/incoherent_control.tex",
        "results/figs/hierarchy_rows.tex",
    ],
    "s2and": [
        "data/s2and/proc/*/genes_*.npz",
        "data/s2and/proc/*/sbert.npz",
        "data/s2and/proc/*/specter.npz",
        "data/s2and/proc/*/instructor.npz",
        "data/s2and/proc/*/paper_text.parquet",
    ],
    "aps": [
        "data/aps/embeddings/*_norm_lora_emb.npz",
        "data/aps/embeddings/*_kron_emb.npz",
        "data/aps/embeddings/*_genkron_emb.npz",
        "data/aps/embeddings/baseline_*.npz",
        "data/aps/embeddings/sbert_allmpnet.npz",
        "data/aps/paper_text.parquet",
    ],
}

# Tiers we actually ship as a Zenodo record. `aps` is inventoried, not bundled.
SHIPPED = ("results", "s2and")

# Files a tier's globs sweep up but that nothing in this workflow produces or reads. The
# `results_*.csv` patterns are deliberately broad (one per eval suffix), so a suffix whose
# chain has been retired has to be named here or a stale file on disk walks back into the
# manifest on the next regeneration.
EXCLUDE: tuple[str, ...] = (
    # ICAE is a decoding comparison in this paper -- it appears in no row of any reported
    # similarity table, and its benchmark-evaluation rules were removed with icae.smk. These
    # CSVs have had no producer and no consumer since; shipping them invites a reader to
    # look for numbers that no table draws on.
    "data/kron/results_*_icae.csv",
    "data/s2and/results_*_icae.csv",
)


def collect(tier: str) -> list[Path]:
    excluded = {p for pattern in EXCLUDE for p in SRC.glob(pattern)}
    seen: list[Path] = []
    for pattern in TIERS[tier]:
        for p in sorted(SRC.glob(pattern)):
            if p.is_file() and p not in excluded and p not in seen:
                seen.append(p)
    return seen


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n}"


def write_manifest(rows: list[tuple[str, str, int, str]]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[tuple[str, str], tuple[str, str, int, str]] = {}
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            tier, path, size, digest = line.split("\t")
            existing[(tier, path)] = (tier, path, int(size), digest)
    for row in rows:
        existing[(row[0], row[1])] = row
    with MANIFEST.open("w") as fh:
        fh.write("# tier\tpath\tbytes\tsha256\n")
        fh.write("# Generated by scripts/make_artifact_bundle.py. Verified by scripts/fetch_artifacts.py.\n")
        for key in sorted(existing):
            tier, path, size, digest = existing[key]
            fh.write(f"{tier}\t{path}\t{size}\t{digest}\n")
    print(f"manifest: {MANIFEST.relative_to(ROOT)} ({len(existing)} rows)")


def make_tar(tier: str, files: list[Path], out_dir: Path, level: int = 10) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    use_zstd = shutil.which("zstd") is not None
    tar_path = out_dir / f"doc2lora-{tier}.tar"
    # dereference: the artifacts are symlinks into a shared store, so the bundle
    # must carry the bytes, not the links.
    with tarfile.open(tar_path, "w", dereference=True) as tf:
        for p in files:
            tf.add(p, arcname=str(p.relative_to(SRC)))
    if use_zstd:
        subprocess.run(["zstd", f"-{level}", "-T0", "--rm", "-f", str(tar_path)], check=True)
        tar_path = tar_path.with_suffix(".tar.zst")
    else:
        subprocess.run(["gzip", "-9", "-f", str(tar_path)], check=True)
        tar_path = tar_path.with_suffix(".tar.gz")
    return tar_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tiers", nargs="*", metavar="TIER",
                    help=f"tiers to bundle: {', '.join(sorted(TIERS))} (shipped: {', '.join(SHIPPED)})")
    ap.add_argument("--out", type=Path, default=ROOT / "dist", help="where the tarballs land (default: dist/)")
    ap.add_argument("--list", action="store_true", help="inventory every tier and exit (no hashing, no tar)")
    ap.add_argument("--manifest-only", action="store_true", help="hash the files and update the manifest, skip the tar")
    ap.add_argument("--root", type=Path, default=None,
                    help="checkout the artifacts live in (default: this repository)")
    ap.add_argument("--level", type=int, default=10,
                    help="zstd level; use 1 for the npz-heavy tiers, they are already compressed")
    args = ap.parse_args()

    global SRC
    if args.root:
        SRC = args.root.resolve()

    if args.list:
        for tier in TIERS:
            files = collect(tier)
            total = sum(p.stat().st_size for p in files)
            missing = "" if files else "   (nothing on disk -- run the workflow first)"
            shipped = "zenodo" if tier in SHIPPED else "NOT distributed (regenerate)"
            print(f"{tier:9s} {len(files):5d} files  {human(total):>9s}  [{shipped}]{missing}")
        return 0

    if not args.tiers:
        ap.error("name at least one tier, or pass --list")
    for tier in args.tiers:
        if tier not in TIERS:
            ap.error(f"unknown tier {tier!r}; choose from {', '.join(sorted(TIERS))}")

    rows: list[tuple[str, str, int, str]] = []
    for tier in args.tiers:
        files = collect(tier)
        if not files:
            print(f"!! {tier}: nothing on disk -- run the workflow first", file=sys.stderr)
            continue
        total = sum(p.stat().st_size for p in files)
        print(f"{tier}: {len(files)} files, {human(total)}")
        for p in files:
            rows.append((tier, str(p.relative_to(SRC)), p.stat().st_size, sha256(p)))
        if not args.manifest_only:
            tar = make_tar(tier, files, args.out, level=args.level)
            print(f"  -> {tar} ({human(tar.stat().st_size)})")
    write_manifest(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
