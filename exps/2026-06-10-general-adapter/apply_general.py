"""[GPU] Apply a trained GENERAL kron adapter to a gene npz -> genkron npz (same key/format).
Usage: CUDA_VISIBLE_DEVICES=0 python apply_general.py <enc> <in_npz> <out_npz> [key=embeddings]
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "exps/2026-06-09-kron-adapter")
from kron import KronAdapter

ENC, IN, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
KEY = sys.argv[4] if len(sys.argv) > 4 else "embeddings"
dev = "cuda"
ck = torch.load(f"exps/2026-06-10-general-adapter/adapter_general_{ENC}.pt", map_location=dev)
L, D = ck["L"], ck["D"]
m = KronAdapter(L=L, d=512).to(dev); m.load_state_dict(ck["state"]); m.eval()

z = np.load(IN, allow_pickle=True)
E = z[KEY].astype(np.float32); N = E.shape[0]
assert E.shape[1] == D, f"dim mismatch {E.shape[1]} vs adapter {D}"
out = np.empty((N, D), np.float16)
with torch.no_grad():
    for i in range(0, N, 8192):
        j = min(N, i + 8192)
        x = torch.tensor(E[i:j], device=dev).view(j - i, L, 512)
        out[i:j] = m(x).half().cpu().numpy()
np.savez(OUT, **{KEY: out, "paper_ids": z["paper_ids"]})
print(f"[saved] {OUT}  ({N} x {D})", flush=True)
