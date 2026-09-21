"""[GPU] Text-embedding baselines for an S2AND sub-dataset (reuses workflow text_encoders registry).
Usage: CUDA_VISIBLE_DEVICES=0 python text_baselines.py zbmath sbert
Output: proc/<ds>/<method>.npz  (paper_ids[str], vecs[N,D] float32)
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, "workflow/scripts")
from text_encoders import get_encoder

DS, METHOD = sys.argv[1], sys.argv[2]
base = "exps/2026-06-09-s2and"
pt = pd.read_parquet(f"{base}/proc/{DS}/paper_text.parquet")
pids = pt.paper_id.astype(str).tolist()
texts = [t if isinstance(t, str) and t.strip() else "." for t in pt.text.tolist()]
print(f"[{DS}/{METHOD}] {len(texts)} papers", flush=True)

enc = get_encoder(METHOD)                      # returns an encode(texts) callable
V = np.asarray(enc(texts), np.float32)
np.savez(f"{base}/proc/{DS}/{METHOD}.npz", paper_ids=np.array(pids), vecs=V)
print(f"[saved] {base}/proc/{DS}/{METHOD}.npz  {V.shape}", flush=True)
