"""Embed the benchmark subsets with gte-large — Text-to-LoRA's native coordinates (#151).

The retrieval half of Arm 1. T2L's hypernetwork reads a FROZEN `gte-large-en-v1.5` vector
and expands it into an adapter, so the honest similarity row for the T2L baseline is that
encoder's own space, reported as **"Text-to-LoRA's native coordinates = gte-large"** and
never as an adapter space. It is not a 2024+-encoder addition on its merits — decision D6
of #142 closed #69 the other way, and gte stays off in `gcb_new_encoders`.

Pooling is CLS in fp32, matching `hyper_llm_modulator.utils.model_loading`'s gte branch
exactly. A different pooling would embed something T2L never sees, which would make the
row a different encoder's result wearing T2L's name.

Embeds only the eval-touched subset (`<field>_eval_ids.parquet`, ~165-250k per field)
rather than the 0.5-1M-paper corpora, the same slicing the rest of the benchmark uses via
BENCH_ROOT. Writes `data/bench/<field>/embeddings/baseline_gte_large.npz` as
{paper_ids, vecs}, the shape `eval_all.py` reads.

  CUDA_VISIBLE_DEVICES=0 python embed_gte_bench.py --field economics
  CUDA_VISIBLE_DEVICES=0 python embed_gte_bench.py --field economics --time-only
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "workflow/scripts"))
os.environ.setdefault("HF_HOME", str(ROOT / "data/agent_assets/hf_cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

IC = ROOT / "exps/2026-06-21-icae-benchmark"
BENCH = ROOT / "data/bench"


def load_texts(field):
    ids = pd.read_parquet(IC / f"{field}_eval_ids.parquet")
    idcol = "paper_id" if "paper_id" in ids.columns else ids.columns[0]
    want = set(int(p) for p in ids[idcol])

    if field == "aps":
        src = ROOT / "data/aps/paper_text_pid.parquet"
        cols = ["paper_id", "title", "abstract"]
    else:
        src = ROOT / f"data/fields/{field}/paper_text.parquet"
        cols = None
    df = pd.read_parquet(src, columns=cols)
    idc = "paper_id" if "paper_id" in df.columns else "aps_paper_id"
    df = df[df[idc].astype(np.int64).isin(want)]
    if "text" in df.columns:
        txt = df["text"].astype(str)
    else:
        txt = (df["title"].fillna("").astype(str).str.strip() + ". "
               + df["abstract"].fillna("").astype(str).str.strip()).str.strip(". ")
    return df[idc].astype(np.int64).to_numpy(), txt.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--time-only", action="store_true",
                    help="embed 512 documents and print the projected full cost")
    a = ap.parse_args()

    from text_encoders import get_encoder

    pids, texts = load_texts(a.field)
    print(f"[data] {a.field}: {len(pids)} eval-touched papers", flush=True)

    enc = get_encoder("gte_large", batch_size=a.batch_size)
    if a.time_only:
        n = min(512, len(texts))
        t0 = time.time()
        enc(texts[:n])
        dt = (time.time() - t0) / n
        print(f"[time] {dt*1000:.1f} ms/doc -> {dt*len(texts)/3600:.2f} h for {a.field}")
        return

    out = BENCH / a.field / "embeddings" / "baseline_gte_large.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    chunk, vecs, t0 = 20000, [], time.time()
    for i in range(0, len(texts), chunk):
        vecs.append(enc(texts[i:i + chunk]))
        done = min(i + chunk, len(texts))
        el = time.time() - t0
        print(f"  {done}/{len(texts)}  {el/60:.1f} min elapsed, "
              f"{el/done*(len(texts)-done)/60:.1f} min left", flush=True)
    V = np.concatenate(vecs, 0).astype(np.float32)
    np.savez(out, paper_ids=pids, vecs=V)
    print(f"wrote {out}  {V.shape}")


if __name__ == "__main__":
    main()
