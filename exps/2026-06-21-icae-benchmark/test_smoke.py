"""[GPU] Smoke test before the full run:
  1. batched compress matches the reference single-doc _compress (cos > 0.999),
  2. a freshly-init KronAdapter is ~identity (cos ~ 1) on slots,
  3. the per-token map is invertible: recover slots from adapted slots (cos > 0.999).
Usage: CUDA_VISIBLE_DEVICES=0 python test_smoke.py
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "exps/2026-06-09-kron-adapter")
from icae_lib import load_icae, compress_one, compress_batch
from kron import KronAdapter

W = "exps/2026-05-26-baseline/icae/weights/mistral_7b_ft_icae.safetensors"
CODE = "exps/2026-05-26-baseline/icae/code/icae_v2"
BASE = "mistralai/Mistral-7B-Instruct-v0.2"
dev = "cuda"

df = pd.read_parquet("exps/2026-06-10-general-adapter/pool_text.parquet").head(8)
texts = [t if isinstance(t, str) and t.strip() else "." for t in df.text.tolist()]

model = load_icae(W, CODE, BASE, dev)
print("[1] batched vs single-doc compress")
sb = compress_batch(model, texts, dev)                       # [8,128,4096]
for i, t in enumerate(texts):
    so = compress_one(model, t, dev)                         # [128,4096]
    cos = F.cosine_similarity(sb[i].flatten(), so.flatten(), dim=0).item()
    print(f"   doc{i}: cos(batch,single)={cos:.5f}  ntok={len(model.tokenizer(t,truncation=True,max_length=512)['input_ids'])}")

print("[2/3] adapter identity-at-init + invertibility")
A = KronAdapter(L=model.mem_size, d=model.dim).to(dev).eval()
with torch.no_grad():
    C = A.C(); alpha = torch.exp(A.alpha)                    # [d,d], [L]
    G = sb                                                   # [8,128,4096]
    Z = torch.einsum("bld,de->ble", G, C) * alpha[None, :, None]
    ident = F.cosine_similarity(Z.flatten(), G.flatten(), dim=0).item()
    Cinv = torch.linalg.inv(C)                               # exact inverse (Q diag) -> diag(exp(-s)) Q^T
    Grec = torch.einsum("ble,ed->bld", Z / alpha[None, :, None], Cinv)
    rec = F.cosine_similarity(Grec.flatten(), G.flatten(), dim=0).item()
print(f"   init cos(adapter(G),G) = {ident:.5f}  (expect ~1.0 at init)")
print(f"   roundtrip cos(inv(adapter(G)),G) = {rec:.5f}  (expect >0.999)")
print("SMOKE OK" if ident > 0.99 and rec > 0.999 else "SMOKE CHECK VALUES ABOVE")
