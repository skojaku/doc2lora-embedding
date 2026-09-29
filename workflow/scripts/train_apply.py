"""Train the Kronecker citation adapter on a field's genes, then apply it and dump a transformed npz.

Usage: CUDA_VISIBLE_DEVICES=1 python workflow/scripts/train_apply.py \
         --field economics --enc gemma --steps 4000 --bs 256 --beta 1e-3
Outputs:
  <emb_dir>/<enc>_kron_emb.npz           (embeddings[N,L*512] fp16, paper_ids)   <- transformed genes
  data/kron/adapter_<field>_<enc>.pt
"""
import argparse, os, sys, time
import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, "workflow/scripts")
sys.path.insert(0, "data/kron")
from bench_data import load, citation_net
from kron import KronAdapter, info_nce

def cit_path(field):
    """APS has its own citation network; fields live under the OpenAlex slices."""
    return citation_net(field)


def load_citation(field, n_nodes):
    z = np.load(cit_path(field), allow_pickle=True)
    A = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    return A


def build_pairs(A, id_to_row, cap=3_000_000, seed=0):
    """All citation edges (anchor->cited) restricted to gene-covered papers, as row-index pairs."""
    A = A.tocoo()
    r, c = A.row, A.col
    ar, pr = id_to_row[r], id_to_row[c]
    keep = (ar >= 0) & (pr >= 0) & (ar != pr)
    ar, pr = ar[keep], pr[keep]
    rng = np.random.default_rng(seed)
    if len(ar) > cap:
        idx = rng.choice(len(ar), cap, replace=False); ar, pr = ar[idx], pr[idx]
    return ar.astype(np.int64), pr.astype(np.int64)


def main(a):
    pt, edges, emb_dir = load(a.field)
    fp = f"{emb_dir}/{a.enc}_norm_lora_emb.npz"
    print(f"[load] genes {fp}", flush=True)
    z = np.load(fp, allow_pickle=True)
    G = torch.from_numpy(z["embeddings"])              # fp16 [N, L*512] on CPU
    pids = z["paper_ids"].astype(np.int64)
    N, D = G.shape; L = D // 512
    print(f"[genes] N={N} D={D} L={L}", flush=True)

    n_nodes = int(pids.max()) + 1
    A = load_citation(a.field, n_nodes)
    id_to_row = np.full(max(n_nodes, A.shape[0]), -1, np.int64)   # cover the citation id space too
    id_to_row[pids] = np.arange(N)
    ar, pr = build_pairs(A, id_to_row[:A.shape[0]], cap=a.cap)
    print(f"[pairs] {len(ar):,} citation training pairs (gene-covered)", flush=True)

    dev = "cuda"
    model = KronAdapter(L=L, d=512).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    Gv = G.view(N, L, 512)
    rng = np.random.default_rng(0)
    t0 = time.time(); ema = None
    for step in range(a.steps):
        sel = rng.integers(0, len(ar), a.bs)
        ga = Gv[ar[sel]].to(dev).float(); gp = Gv[pr[sel]].to(dev).float()
        za, zp = model(ga), model(gp)
        loss = info_nce(za, zp, a.tau) + a.beta * model.logdet_reg()
        opt.zero_grad(); loss.backward(); opt.step()
        ema = loss.item() if ema is None else 0.99 * ema + 0.01 * loss.item()
        if step % 200 == 0 or step == a.steps - 1:
            with torch.no_grad():
                print(f"  step {step:6d}  ema {ema:.4f}  loss {loss.item():.4f}  "
                      f"sum(s)={model.s.sum().item():+.3f} sum(alpha)={model.alpha.sum().item():+.3f} "
                      f"[{time.time()-t0:.0f}s]", flush=True)
    if a.no_save:
        print("[no_save] skipping adapter+npz dump", flush=True); return

    torch.save(model.state_dict(), f"data/kron/adapter_{a.field}_{a.enc}.pt")

    # apply to ALL genes in chunks -> transformed npz
    out = np.empty((N, D), np.float16)
    model.eval()
    with torch.no_grad():
        for i in range(0, N, a.chunk):
            j = min(N, i + a.chunk)
            zt = model(Gv[i:j].to(dev).float())
            out[i:j] = zt.half().cpu().numpy()
    op = f"{emb_dir}/{a.enc}_kron_emb.npz"
    np.savez(op, embeddings=out, paper_ids=pids)
    print(f"[saved] {op}  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--field", default="economics")
    p.add_argument("--enc", default="gemma")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=1e-3)
    p.add_argument("--cap", type=int, default=3_000_000)
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--no_save", action="store_true")
    main(p.parse_args())
