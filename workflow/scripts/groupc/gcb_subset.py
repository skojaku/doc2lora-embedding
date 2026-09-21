"""[CPU] Assemble the benchmark-subset text table for the Group C similarity re-runs (#69/#93/#72).

Encoding a full field corpus (565k-987k papers) with every new encoder is not affordable, and it is
not necessary: data/icae/{field}_eval_ids.parquet (#63) lists exactly the papers
the collab / next-paper / topic harness touches, and slicing to it was verified byte-identical for
evaluation.  Every Group C encoder is run on that subset.

Years come from the SAME table the benchmark harness uses (bench_data.load), not from the OpenAlex
master, because APS ids are not OpenAlex ids.

Out: data/groupc/bench/{field}/subset_text.parquet [paper_id, title, abstract, text, year]
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "workflow/scripts")
sys.path.insert(0, "data/layer_bands")
from bench_data import load  # noqa: E402

FIELD = snakemake.wildcards.field                      # noqa: F821
ids = pd.read_parquet(snakemake.params.eval_ids)       # noqa: F821
ids["paper_id"] = ids.paper_id.astype(np.int64)

txt = pd.read_parquet(snakemake.input.paper_text)      # noqa: F821
txt = txt.rename(columns={"aps_paper_id": "paper_id"})
txt["paper_id"] = txt.paper_id.astype(np.int64)
keep = txt[txt.paper_id.isin(set(ids.paper_id))].copy()

pt, _, _ = load(FIELD)
yr = dict(zip(pt.paper_id.astype(np.int64), pt.year.astype(int)))
keep["year"] = keep.paper_id.map(yr)

for c in ("title", "abstract"):
    if c not in keep.columns:
        keep[c] = ""
keep = keep[["paper_id", "title", "abstract", "text", "year"]]
os.makedirs(os.path.dirname(snakemake.output.subset), exist_ok=True)     # noqa: F821
keep.to_parquet(snakemake.output.subset, index=False)                    # noqa: F821
ymin, ymax = keep.year.min(), keep.year.max()
print(f"[gcb-subset {FIELD}] {len(keep):,} of {len(ids):,} benchmark papers have text "
      f"(years {ymin}-{ymax}, {int(keep.year.isna().sum()):,} missing)", flush=True)
