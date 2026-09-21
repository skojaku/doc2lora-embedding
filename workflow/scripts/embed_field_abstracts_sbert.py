"""[GPU] Full-scale SBERT (all-mpnet) embeddings for every field paper with a usable abstract —
the textual baseline for the field idea-compatibility analysis, on the SAME papers as the
doc2lora genes (the embedding subsample, keyed by field-local paper_id). Sharded + resumable.

Field analogue of embed_abstracts_sbert.py: no PACS1 filter / no aps_paper_table — the field
paper_text.parquet already carries paper_id + abstract.

Dual-mode. Out: npz {paper_ids, vecs}.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd


def _lease_gpu(n=4, lockroot="/tmp/d2l_gpu_locks"):
    """Atomically claim one free GPU (lock-dir) and pin CUDA_VISIBLE_DEVICES to it,
    so up to n single-GPU Snakemake jobs run concurrently on distinct GPUs."""
    import atexit, shutil, time
    os.makedirs(lockroot, exist_ok=True)
    while True:
        for i in range(n):
            lk = os.path.join(lockroot, f"gpu{i}")
            try:
                os.mkdir(lk)
            except FileExistsError:
                try:
                    if os.path.exists(os.path.join(lk, "pid")):
                        os.kill(int(open(os.path.join(lk, "pid")).read()), 0)
                        continue
                except Exception:
                    pass
                shutil.rmtree(lk, ignore_errors=True)
                try:
                    os.mkdir(lk)
                except FileExistsError:
                    continue
            open(os.path.join(lk, "pid"), "w").write(str(os.getpid()))
            atexit.register(lambda l=lk: shutil.rmtree(l, ignore_errors=True))
            os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
            print(f"[gpu-lease] claimed physical GPU {i}", flush=True)
            return i
        time.sleep(5)


def main(paper_text, out_npz, model_name, min_chars=300, shard_dir=None, shard_size=20000):
    _lease_gpu()
    df = pd.read_parquet(paper_text, columns=["paper_id", "abstract"])
    df = df[df["abstract"].str.len().fillna(0) >= min_chars]
    df = df.drop_duplicates("paper_id").sort_values("paper_id").reset_index(drop=True)
    print(f"[sbert] {len(df)} papers (abstract>={min_chars})")

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(model_name)
    shard_dir = shard_dir or (os.path.dirname(out_npz) + "/sbert_shards")
    os.makedirs(shard_dir, exist_ok=True)
    n = len(df)
    nshard = (n + shard_size - 1) // shard_size
    for sidx in range(nshard):
        sp = os.path.join(shard_dir, f"shard_{sidx:04d}.npz")
        if os.path.exists(sp):
            continue
        s, e = sidx * shard_size, min((sidx + 1) * shard_size, n)
        chunk = df.iloc[s:e]
        V = enc.encode(chunk["abstract"].tolist(), normalize_embeddings=True, convert_to_numpy=True,
                       batch_size=256, show_progress_bar=True).astype(np.float32)
        np.savez(sp, paper_ids=chunk["paper_id"].to_numpy(), vecs=V)
        print(f"[sbert] shard {sidx+1}/{nshard} ({s}-{e})")

    ids, vecs = [], []
    for sidx in range(nshard):
        z = np.load(os.path.join(shard_dir, f"shard_{sidx:04d}.npz"))
        ids.append(z["paper_ids"]); vecs.append(z["vecs"])
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez(out_npz, paper_ids=np.concatenate(ids), vecs=np.concatenate(vecs))
    print(f"[sbert] -> {out_npz}  {np.concatenate(vecs).shape}")


if __name__ == "__main__":
    if "snakemake" in sys.modules:
        sm = snakemake  # noqa: F821
        main(sm.input["paper_text"], sm.output["embeddings"], sm.params["model_name"],
             min_chars=int(sm.params["min_chars"]), shard_dir=sm.params["shard_dir"],
             shard_size=int(sm.params["shard_size"]))
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("--paper-text", required=True)
        ap.add_argument("--out-npz", required=True)
        ap.add_argument("--model-name", default="sentence-transformers/all-mpnet-base-v2")
        ap.add_argument("--min-chars", type=int, default=300)
        ap.add_argument("--shard-dir", default=None)
        ap.add_argument("--shard-size", type=int, default=20000)
        a = ap.parse_args()
        main(a.paper_text, a.out_npz, a.model_name, a.min_chars, a.shard_dir, a.shard_size)
