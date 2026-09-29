"""[CPU] Sample the APS papers whose single-document decodes are scored in the #95 fidelity test.

Uniform draw over APS papers that carry a real abstract (>= min_chars), so every row is a
document-scale paper rather than a bare title.

Out: data/groupc/fidelity/sample.parquet [paper_id, title, abstract, text, n_chars]
"""
import os

import numpy as np
import pandas as pd

N = int(snakemake.params.n_papers)          # noqa: F821
SEED = int(snakemake.params.seed)           # noqa: F821
MIN_CHARS = int(snakemake.params.min_chars)  # noqa: F821

df = pd.read_parquet(snakemake.input.paper_text)     # noqa: F821
df = df.rename(columns={"aps_paper_id": "paper_id"})
ok = df.abstract.notna() & (df.abstract.astype(str).str.len() >= MIN_CHARS)
df = df[ok]
print(f"[fid] {len(df):,} APS papers with an abstract of >= {MIN_CHARS} chars", flush=True)

rng = np.random.default_rng(SEED)
take = np.sort(rng.choice(len(df), size=min(N, len(df)), replace=False))
out = df.iloc[take][["paper_id", "title", "abstract", "text"]].copy()
out["paper_id"] = out.paper_id.astype(np.int64)
out["n_chars"] = out.text.astype(str).str.len()
os.makedirs(os.path.dirname(snakemake.output.sample), exist_ok=True)    # noqa: F821
out.to_parquet(snakemake.output.sample, index=False)                    # noqa: F821
print(f"[fid] wrote {snakemake.output.sample}: {len(out)} papers "      # noqa: F821
      f"(median {int(out.n_chars.median())} chars)", flush=True)
