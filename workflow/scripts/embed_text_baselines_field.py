"""[GPU] Embed a field's paper_text.parquet with a NON-doc2lora text baseline.

Baselines for the mobility-compatibility / retrieval comparison against doc2lora
idea-genes. Encoders live in text_encoders.py (shared registry).

Input convention:
  * specter2 -> native (title, abstract) via encode_pairs (title + sep + abstract)
  * others   -> the parquet `text` column (falls back to title+abstract if absent)

Output convention (matches embed_abstracts_sbert.py):
  npz {paper_ids: int64 [N], vecs: float32 [N, D]}  keyed by field-local paper_id.

Dual-mode (Snakemake `snakemake` object OR argparse). Chunked to avoid OOM.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_encoders import PAIR_INPUT_METHODS, get_encoder  # noqa: E402


def _build_text(df):
    """Single-text input for non-pair encoders: prefer `text`, else title+abstract."""
    if "text" in df.columns and df["text"].notna().any():
        s = df["text"].fillna("")
        # rows where text is empty -> backfill from title+abstract
        empty = s.str.len() == 0
        if empty.any():
            bf = (df["title"].fillna("") + ". " + df["abstract"].fillna(""))
            s = s.mask(empty, bf)
        return s.tolist()
    return (df["title"].fillna("") + ". " + df["abstract"].fillna("")).tolist()


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


def main(paper_text, method, output, batch_size=64, gpu=True, chunk_size=20000):
    if gpu:
        # Escape hatch: pin a specific GPU (skip the shared lock-dir lease) by exporting
        # GPU_PIN=<id>. Useful for ad-hoc runs that must avoid busy GPUs the blind lease
        # would otherwise claim. Empty/unset -> normal lock-dir lease across all GPUs.
        pin = os.environ.get("GPU_PIN", "")
        if pin != "":
            os.environ["CUDA_VISIBLE_DEVICES"] = pin
            print(f"[gpu-pin] CUDA_VISIBLE_DEVICES={pin} (lease skipped)", flush=True)
        else:
            _lease_gpu()
    df = pd.read_parquet(paper_text)
    # The field corpora key on `paper_id`; the APS corpus keys on `aps_paper_id`.
    # Accept either so one rule can embed both.
    if "paper_id" not in df.columns:
        if "aps_paper_id" in df.columns:
            df = df.rename(columns={"aps_paper_id": "paper_id"})
        else:
            raise KeyError(f"paper_text {paper_text} lacks a paper_id / aps_paper_id column; has {list(df.columns)}")
    df = df.drop_duplicates("paper_id").sort_values("paper_id").reset_index(drop=True)
    n = len(df)
    print(f"[{method}] {n} papers from {paper_text}", flush=True)

    enc = get_encoder(method, batch_size=batch_size, gpu=gpu)
    if method in PAIR_INPUT_METHODS:
        print(f"[{method}] adapter_path={getattr(enc, 'adapter_path', 'n/a')}", flush=True)

    is_pair = method in PAIR_INPUT_METHODS
    if not is_pair:
        texts = _build_text(df)

    vecs = []
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        chunk = df.iloc[s:e]
        if is_pair:
            V = enc.encode_pairs(chunk["title"].fillna("").tolist(),
                                 chunk["abstract"].fillna("").tolist())
        else:
            V = enc(texts[s:e])
        vecs.append(np.asarray(V, dtype=np.float32))
        print(f"[{method}] chunk {s}-{e} -> {V.shape}", flush=True)

    V = np.concatenate(vecs, axis=0).astype(np.float32)
    paper_ids = df["paper_id"].to_numpy(dtype=np.int64)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    np.savez(output, paper_ids=paper_ids, vecs=V)
    print(f"[{method}] -> {output}  paper_ids{paper_ids.shape} vecs{V.shape}", flush=True)


if __name__ == "__main__":
    if "snakemake" in sys.modules:
        sm = snakemake  # noqa: F821
        main(sm.input["paper_text"], sm.params["method"], sm.output["embeddings"],
             batch_size=int(sm.params.get("batch_size", 64)),
             gpu=bool(sm.params.get("gpu", True)),
             chunk_size=int(sm.params.get("chunk_size", 20000)))
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("--paper-text", required=True)
        ap.add_argument("--method", required=True)
        ap.add_argument("--output", required=True)
        ap.add_argument("--batch-size", type=int, default=64)
        ap.add_argument("--gpu", action="store_true", default=False)
        ap.add_argument("--chunk-size", type=int, default=20000)
        a = ap.parse_args()
        main(a.paper_text, a.method, a.output, batch_size=a.batch_size, gpu=a.gpu,
             chunk_size=a.chunk_size)
