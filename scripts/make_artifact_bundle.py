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
import os
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
    # Everything `snakemake paper_assets` reads but cannot make on a CPU: the outputs of
    # the GPU decodes and extractions, of the steps that need the licensed APS text, and
    # of the judge panel. The rules that turn these into the manuscript's tables and
    # figures are NOT short-circuited: their outputs are excluded below (EXCLUDE), so an
    # unpacked tier leaves exactly the CPU table/figure rules for Snakemake to run.
    "results": [
        # Tab. similarity, Tab. encoder-matrix, Tab. collab-per-window: the paired per-unit
        # score pools that unc_bootstrap resamples (the pools need the 644k gene matrices).
        "data/uncertainty/pools/*.parquet",
        # per-task score tables behind numbers typed into the text
        "data/kron/results_*.csv",
        "data/s2and/results_*.csv",
        "data/s2and/proc/*/sig_table.parquet",
        # invertible adapters (small: ~10^5-10^7 params) and the citation sample of g_theta
        "data/kron/adapter_*.pt",
        "data/general_adapter/adapter_general_*.pt",
        "data/general_adapter/leakage_overlap.json",
        "data/general_adapter/triplets_1x.parquet",
        "data/general_adapter/triplets.parquet",
        # the PACS node set the cluster labels are scored against
        "data/pacs/results/*.parquet",
        "data/pacs/results/tree.json",
        # Tab. hierarchy-labels + Tab. label-eval: every method's raw cluster labels (GPU
        # decodes, vec2text, KeyLLM, ICAE, in-context, T2L, BERTopic), the node set, the
        # node means, the judge verdicts, and the Wikipedia radius walk (Fig. cluster-labels).
        "data/labels/*.json",
        "data/labels/length_vs_breadth.csv",
        "data/labels/qwen_fullrank_means.npz",
        # Tab. prompt-sensitivity: the paraphrase decodes
        "data/groupc/psens/psens_*.json",
        # Tab. incoherent-control: clusters, decodes, and the SBERT + judge scores
        "data/groupc/incoherent/*.json",
        "data/groupc/incoherent/*.parquet",
        "data/groupc/fidelity/sample.parquet",
        # App. symmetric-adapter: the per-unit score pools of the symmetric-adapter benchmark
        "data/groupc/bench/pools/*",
        # Fig. cluster-labels (e),(f), Tab. mixing-decode, Fig. psens-edge: the pair-axis
        # decodes (Doc2LoRA and ICAE, every stratum and paraphrase), their corner specs,
        # and the SBERT landing-position / copy-rate / T2L caches the figure reads
        "data/pair_axis/corners_pair*.json",
        "data/pair_axis/pair_manifest.json",
        "data/pair_axis/results/absfollow_pair*_pairaxis.json",
        "data/pair_axis/results/absfollow_pair*_pairaxis_icae.json",
        "data/pair_axis/results/absfollow_pair*_pairaxis_p?.json",
        "data/pair_axis/results/absfollow_pair*_pairaxis_p?_icae.json",
        "data/pair_axis/results/colorband_*_pairaxis.json",
        "data/pair_axis/results/pair_axis_*_pairaxis*.json",
        "data/t2l/results/midpoints_t2l_*.json",
        # benchmark pair lists + the pooling rho quoted in Sec. methods
        "data/collab_scores_aps_*.parquet",
        "data/aps/pooling_spearman.csv",
        # the manuscript's own tables, to diff a rebuild against (results/figs is where
        # the rebuild lands, so the reference copies live apart from it)
        "data/reference/*.tex",
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

# Member timestamp (2026-09-01 00:00 UTC). Override with SOURCE_DATE_EPOCH.
MTIME = int(os.environ.get("SOURCE_DATE_EPOCH", 1788220800))

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
    # Outputs of the CPU rules `paper_assets` runs. Shipping them would let Snakemake skip
    # the very steps a reader unpacks the tier to re-run.
    "data/labels/label_eval_nodes.json",
    "data/labels/label_eval_metric1.json",
    "data/labels/label_eval_summary.json",
    "data/groupc/psens/psens_scores.json",
    # hand-kept backups next to the live label files
    "data/labels/*.bak.json",
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
    # A tier that was just bundled replaces its rows wholesale, so a file that left the
    # tier also leaves the manifest. Tiers not bundled in this call keep their rows.
    rebuilt = {row[0] for row in rows}
    existing = {k: v for k, v in existing.items() if k[0] not in rebuilt}
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
    # One timestamp for every member. The archive then does not depend on when a file
    # happened to be written in the source run, and -- what matters for Snakemake -- no
    # unpacked input is newer than an unpacked output, so `--rerun-triggers mtime` never
    # schedules a rule whose result ships in the tier.
    def normalise(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.mtime = MTIME
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        return info

    with tarfile.open(tar_path, "w", dereference=True, format=tarfile.PAX_FORMAT) as tf:
        for p in sorted(files):
            tf.add(p, arcname=str(p.relative_to(SRC)), filter=normalise)
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
