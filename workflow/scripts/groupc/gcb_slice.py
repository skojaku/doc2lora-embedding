"""[CPU] Slice a cached full-corpus embedding npz down to a Group C benchmark subset (#145).

Some baselines were already embedded over the whole field corpus, so the Group C subset does not
need a fresh GPU pass -- it needs the same vectors restricted to the papers the harness touches.
That is strictly better than re-embedding: the rows are the ones the manuscript's own table uses.

Usage: python workflow/scripts/groupc/gcb_slice.py --src ... --subset ... --out ...
"""
import argparse
import os

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="full-corpus npz (paper_ids + vecs/embeddings)")
    p.add_argument("--subset", required=True, help="subset_text.parquet with the wanted paper_id")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    want = pd.read_parquet(a.subset, columns=["paper_id"]).paper_id.astype(np.int64)
    z = np.load(a.src, allow_pickle=True)
    key = "embeddings" if "embeddings" in z.files else "vecs"
    ids = z["paper_ids"].astype(np.int64)
    row = {int(p): i for i, p in enumerate(ids)}
    missing = [int(p) for p in want if int(p) not in row]
    if missing:
        raise SystemExit(f"[gcb-slice] {len(missing):,} of {len(want):,} subset papers absent from "
                         f"{a.src} (first: {missing[:5]}); the slice would silently drop them.")
    keep = np.array([row[int(p)] for p in want], dtype=np.int64)
    V = np.asarray(z[key][keep], dtype=np.float32)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, vecs=V, paper_ids=want.values)
    print(f"[gcb-slice] {a.src} -> {a.out}  {V.shape}", flush=True)


if __name__ == "__main__":
    main()
