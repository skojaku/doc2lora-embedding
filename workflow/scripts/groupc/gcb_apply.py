"""[GPU] Apply a trained Kron transform to an embedding npz (#93/#72).

Usage: python workflow/scripts/groupc/gcb_apply.py --adapter ... --in ... --out ...
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "exps/2026-06-09-kron-adapter")
from kron import KronAdapter  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.adapter, map_location=dev)
    L, D = int(ck["L"]), int(ck["D"])
    m = KronAdapter(L=L, d=D // L).to(dev)
    m.load_state_dict(ck["state"])
    m.eval()

    z = np.load(a.inp, allow_pickle=True)
    key = "embeddings" if "embeddings" in z.files else "vecs"
    E = np.asarray(z[key], dtype=np.float32).reshape(z[key].shape[0], -1)
    assert E.shape[1] == D, f"dim mismatch {E.shape[1]} vs adapter {D}"
    out = np.empty((len(E), D), np.float16)
    with torch.no_grad():
        for i in range(0, len(E), 8192):
            jx = min(len(E), i + 8192)
            x = torch.tensor(E[i:jx], device=dev).view(jx - i, L, D // L)
            out[i:jx] = m(x).half().cpu().numpy()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, vecs=out, paper_ids=z["paper_ids" if "paper_ids" in z.files else "pids"])
    print(f"[saved] {a.out}  {out.shape}", flush=True)


if __name__ == "__main__":
    main()
