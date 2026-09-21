"""Fairness control: put the SAME citation-trained invertible adapter on top of a flat TEXT embedding.
For a d-dim text vector the Kron adapter reduces to C=Q*diag(exp s) (d x d, rotation x positive diag),
A/B drop out (no layer/query axes). Trained with the same in-batch InfoNCE on citation edges.

Usage: CUDA_VISIBLE_DEVICES=0 python exps/2026-06-09-kron-adapter/train_text_kron.py --field economics --enc sbert --steps 8000
Saves <emb_dir>/<enc>_kron.npz  (vecs[N,d] fp16, paper_ids).
"""
import argparse, sys, time
import numpy as np
import torch

sys.path.insert(0, "workflow/scripts"); sys.path.insert(0, "exps/2026-06-09-kron-adapter")
from bench_data import load
from kron import KronAdapter, info_nce
from train_apply import load_citation, build_pairs

FILES = {"sbert": ("sbert_allmpnet.npz", "vecs"), "specter2": ("baseline_specter2.npz", "vecs"),
         "instructor": ("baseline_instructor.npz", "vecs")}


def main(a):
    pt, edges, emb_dir = load(a.field)
    fn, key = FILES[a.enc]
    z = np.load(f"{emb_dir}/{fn}", allow_pickle=True)
    G = torch.from_numpy(z[key].astype(np.float16)); pids = z["paper_ids"].astype(np.int64)
    N, D = G.shape
    print(f"[text] {a.enc} N={N} D={D}", flush=True)
    A = load_citation(a.field, N)
    id_to_row = np.full(max(int(pids.max()) + 1, A.shape[0]), -1, np.int64); id_to_row[pids] = np.arange(N)
    ar, pr = build_pairs(A, id_to_row[:A.shape[0]], cap=a.cap)
    print(f"[pairs] {len(ar):,}", flush=True)

    dev = "cuda"
    model = KronAdapter(L=1, d=D).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    Gv = G.view(N, 1, D); rng = np.random.default_rng(0); t0 = time.time(); ema = None
    for step in range(a.steps):
        sel = rng.integers(0, len(ar), a.bs)
        za = model(Gv[ar[sel]].to(dev).float()); zp = model(Gv[pr[sel]].to(dev).float())
        loss = info_nce(za, zp, a.tau) + a.beta * model.logdet_reg()
        opt.zero_grad(); loss.backward(); opt.step()
        ema = loss.item() if ema is None else 0.99 * ema + 0.01 * loss.item()
        if step % 500 == 0 or step == a.steps - 1:
            print(f"  step {step:5d}  ema {ema:.4f}  [{time.time()-t0:.0f}s]", flush=True)

    out = np.empty((N, D), np.float16)
    model.eval()
    with torch.no_grad():
        for i in range(0, N, a.chunk):
            j = min(N, i + a.chunk)
            out[i:j] = model(Gv[i:j].to(dev).float()).half().cpu().numpy()
    op = f"{emb_dir}/{a.enc}_kron.npz"
    np.savez(op, vecs=out, paper_ids=pids)
    print(f"[saved] {op} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--field", default="economics"); p.add_argument("--enc", default="sbert")
    p.add_argument("--steps", type=int, default=8000); p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=1e-3); p.add_argument("--cap", type=int, default=3_000_000)
    p.add_argument("--chunk", type=int, default=16384)
    main(p.parse_args())
