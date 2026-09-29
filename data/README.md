# data/

Everything under this directory is fetched or generated — nothing but the
manifest is tracked.

- `ARTIFACTS.tsv` — the Zenodo manifest: one row per archived file
  (`tier`, `path`, `bytes`, `sha256`). `scripts/fetch_artifacts.py` verifies
  downloads against it; `scripts/make_artifact_bundle.py` regenerates it.
- Everything else: point `workflow/config.yaml:data_dir` here (or symlink this
  directory at your storage) and let the workflow fill it.

Tiers: `results` (~192 MB, 219 files — rebuild every table on a CPU),
`s2and` (~12.3 GB, 50 files — re-score name disambiguation from the vectors),
`aps` (~137 GB, 12 files — NOT distributed; `snakemake all_embeddings`).
