# Uncertainty: per-unit score pools for bootstrap CIs

All headline benchmark numbers are single-seed point estimates. This dir dumps the **per-unit evaluation
scores** (paired across methods — same units for every method) so confidence intervals / std can be computed
by bootstrap. `bootstrap.py` is a reference implementation; another agent can resample these pools directly.

Methods compared (per config): `{enc}` (raw gene), `{enc}_kron` (per-field citation / same-author adapter),
`{enc}_genkron` (the general master-OpenAlex adapter), `sbert`, `specter2`/`specter`, `instructor`.
Encoders: gemma / qwen / mistral.

## Pool files (`pools/`) and bootstrap recipe

| file | rows = one ... | columns | metric | bootstrap unit |
|------|----------------|---------|--------|----------------|
| `collab_<field>_<enc>.parquet` | candidate author-pair | `window, a, b, y`, + one score col per method (cosine of author centroids) | ROC-AUC of `(y, score)` | resample rows |
| `np_<field>_<enc>.parquet` | (query × candidate) | `query_idx, cand_pid, is_pos`, + one score col per method | pooled ROC-AUC of `(is_pos, score)` | **cluster-resample `query_idx`** |
| `topic_<field>_<enc>.parquet` | test paper | `paper_id, true`, + one predicted-label col per method (cosine-kNN) | macro-F1 `(true, pred)` | resample rows |
| `s2and_<ds>_<enc>.parquet` | test signature | `dataset, enc, method, block, signature_id, b3_p, b3_r` | B³-F1 = `2·mean(P)·mean(R)/(mean(P)+mean(R))` | resample `signature_id` (or `block`) |

Notes:
- Field pools name method columns by encoder (`gemma`, `gemma_kron`, `gemma_genkron`, `sbert`, ...). S2AND
  uses generic method names (`gene`, `gene_kron`, `gene_genkron`, `specter`, `sbert`, `instructor`) in a
  `method` column (long format).
- `np` units are **paired** across methods via `query_idx`; do a paired (same resampled units) bootstrap to
  test small differences. `collab`/`topic` rows are paired by row; `s2and` by `signature_id`.
- B³-F1 is a ratio of means, so bootstrap P and R together (resample signatures, take mean P & mean R, then F).

## Generate / refresh

Wired into the main Snakemake workflow (`workflow/rules/uncertainty.smk`). The DAG builds any missing
gene/kron/genkron/text embeddings via the `kron`, `general`, `fields`, `baselines`, `s2and` sub-workflows,
then the pools, then the bootstrap. Pass `--rerun-triggers mtime` (like `kron`) so it does NOT re-extract genes:

```bash
snakemake uncertainty --rerun-triggers mtime -j4      # all 42 pools + uncertainty_summary.csv + replicates
# tune scope/effort via --config, e.g.:
#   --config unc_nboot=2000
#   --config unc_s2and='["zbmath","qian"]'
```

Targets: `unc_pool_field` (collab/np/topic per field×enc), `unc_pool_s2and` (per ds×enc), `unc_bootstrap`
(reads every pool → summary + replicates). `uncertainty` is the top-level alias.

Run a single piece directly (what the rules call under the hood):

```bash
CUDA_VISIBLE_DEVICES="" python score_pool_field.py <field> <enc>     # economics|psychology|aps × gemma|qwen|mistral
CUDA_VISIBLE_DEVICES="" python score_pool_s2and.py <dataset> <enc>   # zbmath|qian|arnetminer|pubmed|kisti × enc
python bootstrap.py [N_BOOT=1000]   # -> bootstrap_replicates.parquet (long) + uncertainty_summary.csv
```

`finish_mistral.sh` is the one-shot manual driver used to complete the mistral econ/psych rows (kron→genkron
→pools→bootstrap on a single pinned GPU); the Snakemake target now supersedes it for routine runs.

`uncertainty_summary.csv` columns: `benchmark, group(field/dataset), enc, method, metric, mean, std,
ci2.5, ci97.5`. `bootstrap_replicates.parquet` is long (`...,boot,value`) for violins/CDFs.

Pool means reproduce the reported point estimates exactly (validated on economics·gemma and qian·gemma).
mistral·economics / mistral·psychology pools are produced once those field genes finish extracting.
