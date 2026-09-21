"""[CPU] Numeric proof that the bench subset reproduces the full-data eval EXACTLY: run collab + next-
paper for one encoder on full vs bench, same protocol (fixed cohorts + fixed negative pool). Both must
match to all printed digits. (topic uses GPU; it is covered by the byte-identity check in verify_subset.)

Usage: python verify_eval_cpu.py economics qwen
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, "workflow/scripts")
sys.path.insert(0, "exps/2026-06-07-layer-bands")
sys.path.insert(0, "exps/2026-06-09-kron-adapter")
from bench_data import load

field = sys.argv[1]
enc = sys.argv[2] if len(sys.argv) > 2 else "qwen"
IC = "exps/2026-06-21-icae-benchmark"
os.environ["NP_FUT_POOL_FILE"] = f"{IC}/{field}_fut_pool.parquet"
os.environ["NP_COHORT_FILE"] = f"{IC}/{field}_cohorts.parquet"

import importlib.util
spec = importlib.util.spec_from_file_location("ea", "exps/2026-06-09-kron-adapter/eval_all.py")
ea = importlib.util.module_from_spec(spec); spec.loader.exec_module(ea)

pt, edges, emb_dir = load(field)
yr = dict(zip(pt.paper_id.astype(int), pt.year)); edges = edges.copy()
edges["year"] = edges.paper_id.map(yr); edges = edges.dropna(subset=["year"]); edges["year"] = edges.year.astype(int)
dcol = pd.read_parquet(f"data/collab_scores_{field}_hard.parquet").reset_index(drop=True)
fn, key = f"{enc}_norm_lora_emb.npz", "embeddings"

print(f"[{field}/{enc}] collab + next-paper, full vs bench (fixed cohorts + negative pool)")
res = {}
for label, ed in [("full", emb_dir), ("bench", f"data/bench/{field}/embeddings")]:
    cau, cap = ea.collab_eval(field, ed, edges, enc, fn, key, True, dcol)
    npauc = ea.np_eval(field, ed, edges, fn, key)
    res[label] = (cau, npauc)
    print(f"  {label:6} collab_AUC={cau:.8f}  np_AUC={npauc:.8f}", flush=True)
ok = np.allclose(res["full"], res["bench"], atol=0, rtol=0)
print("IDENTICAL" if ok else f"DIFFER: full={res['full']} bench={res['bench']}")
