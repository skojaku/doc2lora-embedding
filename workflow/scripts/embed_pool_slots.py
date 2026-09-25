"""[GPU] ICAE-compress the OpenAlex adapter-training pool (pool_text.parquet, ~167k papers) into
memory slots and store them as a fp16 memmap [N, 128, 4096]. These per-token slots are what the
invertible per-token adapter is trained on (train_icae_adapter.py). ~167GB on disk; transient.

Usage: CUDA_VISIBLE_DEVICES=0 BATCH=24 python embed_pool_slots.py
Outputs: pool_slots.npy (memmap fp16 [N,128,4096]), pool_pids.npy (int64 [N])
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
import numpy.lib.format as fmt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from icae_lib import load_icae, compress_batch

GA = "data/general_adapter"
from bench_data import path as cfg_path   # bench_data.py sits next to this file
W = cfg_path("icae_weights")
CODE = cfg_path("icae_code_dir")
BASE = "mistralai/Mistral-7B-Instruct-v0.2"
BATCH = int(os.environ.get("BATCH", "24"))
dev = "cuda"

# Shard across GPUs: launch NSHARD copies (SHARD=0..NSHARD-1), each writes a disjoint row range of
# the SAME pre-allocated memmap. Shard 0 creates the file + saves pids; others wait for it to appear.
SHARD = int(os.environ.get("SHARD", "0"))
NSHARD = int(os.environ.get("NSHARD", "1"))

df = pd.read_parquet(f"{GA}/pool_text.parquet")
pids = df.pid.to_numpy(np.int64)
texts = [t if isinstance(t, str) and t.strip() else "." for t in df.text.tolist()]
N = len(texts)

model = load_icae(W, CODE, BASE, dev)
mem, d = model.mem_size, model.dim
path = f"{HERE}/pool_slots.npy"
if SHARD == 0:
    slots = fmt.open_memmap(path, mode="w+", dtype=np.float16, shape=(N, mem, d))
    np.save(f"{HERE}/pool_pids.npy", pids)
else:
    while not os.path.exists(path):
        time.sleep(5)
    time.sleep(10)
    slots = fmt.open_memmap(path, mode="r+")

lo = (N * SHARD) // NSHARD
hi = (N * (SHARD + 1)) // NSHARD
print(f"[pool] shard {SHARD}/{NSHARD} rows [{lo:,},{hi:,}) of {N:,} (batch={BATCH})", flush=True)

# length-bucketing within this shard: group similar-length docs to cut left-padding waste (~2x).
rows = sorted(range(lo, hi), key=lambda k: len(texts[k]))
t0 = time.time(); done = 0
for bi in range(0, len(rows), BATCH):
    idx = rows[bi:bi + BATCH]
    s = compress_batch(model, [texts[k] for k in idx], dev)   # [b,128,4096] f32
    sh = s.half().cpu().numpy()
    for j, k in enumerate(idx):
        slots[k] = sh[j]
    done += len(idx)
    if bi % (BATCH * 100) == 0:
        el = time.time() - t0
        print(f"  shard{SHARD} {done:,}/{hi-lo:,}  {el:.0f}s  {done/max(el,1):.0f} doc/s", flush=True)
slots.flush()
print(f"[saved] {HERE}/pool_slots.npy ({N},{mem},{d})  ({time.time()-t0:.0f}s)", flush=True)
