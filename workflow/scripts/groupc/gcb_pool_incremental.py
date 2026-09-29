"""[GPU] Build the g_theta training pool for a DIFFERENT triplet sample, reusing what is embedded.

The 2x resample of 2026-06-25 drew a fresh sample instead of extending the original one, so the 2x
pool (335,085 papers) holds only 82,635 of the 167,111 papers the reported 1x transform was trained
on. Switching the control back to the 1x tuples therefore needs the missing half embedded, not the
whole pool re-embedded. This script embeds only the papers absent from an existing pool npz and
writes the union restricted to the requested pool, in that pool's own id order.

Usage:
  python workflow/scripts/groupc/gcb_pool_incremental.py --method sbert \
      --pool data/general_adapter/pool_text_1x.parquet \
      --existing data/groupc/bench/pool/pool_sbert.npz \
      --out data/groupc/bench/pool/pool_sbert_1x.npz
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
    p.add_argument("--method", required=True)
    p.add_argument("--pool", required=True, help="parquet of the wanted pool (pid, text)")
    p.add_argument("--existing", default="", help="npz whose rows can be reused")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--chunk-size", type=int, default=20000)
    a = p.parse_args()

    df = pd.read_parquet(a.pool)
    idc = "pid" if "pid" in df.columns else "paper_id"
    ids = df[idc].astype(np.int64).values
    texts = df.text.astype(str).tolist()

    reuse = {}
    dim = None
    if a.existing and os.path.exists(a.existing):
        z = np.load(a.existing, allow_pickle=True)
        key = "vecs" if "vecs" in z.files else "embeddings"
        V = z[key]
        dim = V.shape[1]
        have = {int(p): i for i, p in enumerate(z["paper_ids"])}
        reuse = {pid: V[have[int(pid)]] for pid in ids if int(pid) in have}
    todo = [i for i, pid in enumerate(ids) if int(pid) not in reuse]
    print(f"[pool-inc {a.method}] want {len(ids):,}; reuse {len(reuse):,}; embed {len(todo):,}",
          flush=True)

    fresh = {}
    if todo:
        enc = get_encoder(a.method, batch_size=a.batch_size, gpu=True)
        t0 = time.time()
        for c0 in range(0, len(todo), a.chunk_size):
            block = todo[c0:c0 + a.chunk_size]
            V = np.asarray(enc([texts[i] for i in block]), dtype=np.float32)
            for i, v in zip(block, V):
                fresh[int(ids[i])] = v
            print(f"  {min(c0 + a.chunk_size, len(todo)):,}/{len(todo):,}  "
                  f"[{time.time() - t0:.0f}s]", flush=True)
        dim = dim or V.shape[1]

    out = np.empty((len(ids), dim), np.float32)
    for k, pid in enumerate(ids):
        out[k] = reuse.get(int(pid), fresh.get(int(pid)))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, vecs=out, paper_ids=ids)
    print(f"[saved] {a.out}  {out.shape}", flush=True)


if __name__ == "__main__":
    main()
