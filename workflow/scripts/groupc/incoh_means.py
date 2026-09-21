"""[GPU] Full-rank Doc2LoRA centroid for each cluster of the #101 control set.

Both arms go through the identical extraction path (workflow/scripts/groupc/d2l.full_mean), so the
only difference between a real PACS node and its control is which papers are in it.

Usage: NEED_MB=22000 bash workflow/scripts/gpu_lease.sh \
         python workflow/scripts/groupc/incoh_means.py --clusters ... --arm control \
           --shard 0 --nshards 4 --out data/groupc/incoherent/means_control_s0.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d2l  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clusters", required=True)
    p.add_argument("--arm", choices=["real", "control"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--paper_text", default="data/aps/paper_text.parquet")
    p.add_argument("--max_batch_tokens", type=int, default=6144)
    p.add_argument("--chunk", type=int, default=256)
    a = p.parse_args()

    spec = json.load(open(a.clusters))[a.arm]
    spec = spec[a.shard::a.nshards]
    key = "node" if a.arm == "real" else "cid"
    txt = d2l.aps_texts(a.paper_text)
    print(f"[incoh-means {a.arm} {a.shard}/{a.nshards}] {len(spec)} clusters", flush=True)

    model, gen_tok, ctx_tok = d2l.load(a.ckpt, mode="embed")

    store, ids, ns = [], [], []
    # resumable: keep whatever this shard already wrote
    if os.path.exists(a.out):
        z = np.load(a.out, allow_pickle=True)
        for i, k in enumerate(z["ids"]):
            store.append(z["means"][i]); ids.append(str(k)); ns.append(int(z["n_used"][i]))
        print(f"  resume: {len(ids)} done", flush=True)

    def save():
        np.savez(a.out, means=np.stack(store).astype(np.float32),
                 ids=np.array(ids), n_used=np.array(ns))

    t0 = time.time()
    for c in spec:
        cid = str(c[key])
        if cid in ids:
            continue
        texts = [txt[int(p)] for p in c["members"] if int(p) in txt]
        m = d2l.full_mean(model, ctx_tok, texts, chunk=a.chunk,
                          max_batch_tokens=a.max_batch_tokens)
        store.append(m); ids.append(cid); ns.append(len(texts))
        save()
        print(f"  [{cid}] n={len(texts):,} ||mean||={np.linalg.norm(m):.3f} "
              f"[{time.time() - t0:.0f}s]", flush=True)
    if store:
        save()
    print(f"[saved] {a.out} ({len(ids)} clusters, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
