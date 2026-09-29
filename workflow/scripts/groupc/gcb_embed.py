"""[GPU] Embed a parquet of documents with one encoder from the shared registry (#69/#93).

Works for both the benchmark subsets and the g_theta training pool; the id column may be `paper_id`
or `pid`.  Encoders come from workflow/scripts/text_encoders.py, so the new 2024-era baselines are
the same objects the rest of the workflow uses.

Usage: python workflow/scripts/groupc/gcb_embed.py --input ... --method bge --out ...
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "workflow/scripts")
from text_encoders import get_encoder  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--chunk-size", type=int, default=20000)
    p.add_argument("--shard", default="", help="i/n: embed only shard i of n (contiguous split). "
                                               "Shards are merged by whoever consumes them; ids are kept.")
    a = p.parse_args()

    df = pd.read_parquet(a.input)
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        bounds = np.linspace(0, len(df), n + 1).astype(int)
        df = df.iloc[bounds[i]:bounds[i + 1]].reset_index(drop=True)
        print(f"[gcb-embed] shard {i}/{n}: rows {bounds[i]:,}-{bounds[i + 1]:,}", flush=True)
    idc = "paper_id" if "paper_id" in df.columns else "pid"
    ids = df[idc].astype(np.int64).values
    texts = df.text.astype(str).tolist()
    print(f"[gcb-embed {a.method}] {len(texts):,} docs from {a.input}", flush=True)

    enc = get_encoder(a.method, batch_size=a.batch_size, gpu=True)
    out, t0 = [], time.time()
    for c0 in range(0, len(texts), a.chunk_size):
        out.append(np.asarray(enc(texts[c0:c0 + a.chunk_size]), dtype=np.float32))
        print(f"  {min(c0 + a.chunk_size, len(texts)):,}/{len(texts):,}  "
              f"[{time.time() - t0:.0f}s]", flush=True)
    V = np.concatenate(out)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, vecs=V.astype(np.float32), paper_ids=ids)
    print(f"[saved] {a.out}  {V.shape}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
