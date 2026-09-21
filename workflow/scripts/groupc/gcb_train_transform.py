"""[GPU] Train the SAME bijective Kron transform on the SAME citation tuples, in any embedding space.

This is the control #93 asks for: the manuscript's *adapted* Doc2LoRA column is a map trained on
OpenAlex citation tuples with InfoNCE and in-batch negatives, while every text baseline is used off
the shelf with no equivalent stage.  Here SBERT / Instructor / EmbeddingGemma / BGE / GTE get the
identical treatment (same triplets, same loss, same steps, same optimiser), and so does the gene
space itself, so the table can be read as like-for-like.

It is also the #72 variant: `--cutoff 2018` keeps only triplets whose four papers are ALL pre-cutoff,
giving a temporally disjoint transform to evaluate on post-cutoff papers.

The parameter count follows the space: the gene space factorises as 36 layers x 512 channels
(262,180 params); a flat 768-dim text space gets one 768x768 channel map (590,593).  Both are exactly
invertible, which is the property the paper needs.

Usage: python workflow/scripts/groupc/gcb_train_transform.py --pool ... --triplets ... --out ...
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, "exps/2026-06-09-kron-adapter")
from kron import KronAdapter  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True, help="npz with the pool embeddings")
    p.add_argument("--triplets", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--key", default=None, help="npz key (default: embeddings if present else vecs)")
    p.add_argument("--id-key", default=None, help="npz id key (default: pids if present else paper_ids)")
    p.add_argument("--layers", type=int, default=1, help="L for the Kron factorisation (36 for genes)")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--cutoff", type=int, default=0, help="keep triplets with all years < cutoff")
    p.add_argument("--year-table", default="", help="csv/parquet with paper_id,year (for --cutoff)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    z = np.load(a.pool, allow_pickle=True)
    key = a.key or ("embeddings" if "embeddings" in z.files else "vecs")
    idk = a.id_key or ("pids" if "pids" in z.files else "paper_ids")
    G = z[key]
    ids = {int(k): i for i, k in enumerate(z[idk])}
    D = int(np.prod(G.shape[1:]))
    L = int(a.layers)
    assert D % L == 0, f"dim {D} not divisible by L={L}"
    d = D // L

    T = pd.read_parquet(a.triplets)
    ok = T.a.isin(ids) & T.pos.isin(ids) & T.neg_easy.isin(ids) & T.neg_hard.isin(ids)
    T = T[ok]
    n_before = len(T)
    if a.cutoff:
        if not a.year_table:
            raise SystemExit("--cutoff needs --year-table")
        yt = (pd.read_parquet(a.year_table) if a.year_table.endswith(".parquet")
              else pd.read_csv(a.year_table, usecols=["paper_id", "year"]))
        yr = dict(zip(yt.paper_id.astype(np.int64), pd.to_numeric(yt.year, errors="coerce")))
        def old(col):
            v = T[col].map(yr)
            return v.notna() & (v < a.cutoff)
        T = T[old("a") & old("pos") & old("neg_easy") & old("neg_hard")]
    T = T.reset_index(drop=True)
    print(f"[gcb-train] pool {G.shape} L={L} d={d}; triplets {len(T):,} "
          f"(of {n_before:,} covered{f', cutoff <{a.cutoff}' if a.cutoff else ''})", flush=True)
    if len(T) < 1000:
        raise SystemExit("too few triplets after filtering")

    ia = T.a.map(ids).values; ip = T.pos.map(ids).values
    ie = T.neg_easy.map(ids).values; ih = T.neg_hard.map(ids).values
    Gt = torch.from_numpy(np.asarray(G).reshape(G.shape[0], D))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = KronAdapter(L=L, d=d).to(dev)
    n_params = sum(q.numel() for q in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    rng = np.random.default_rng(a.seed)
    ema, t0 = None, time.time()

    def emb(idx):
        return model(Gt[idx].view(len(idx), L, d).float().to(dev))

    for step in range(a.steps):
        sel = rng.integers(0, len(T), a.batch)
        za = F.normalize(emb(ia[sel]), dim=1)
        zp = F.normalize(emb(ip[sel]), dim=1)
        zh = F.normalize(emb(ih[sel]), dim=1)
        ze = F.normalize(emb(ie[sel]), dim=1)
        logits = torch.cat([za @ zp.t() / a.tau,
                            (za * zh).sum(1, keepdim=True) / a.tau,
                            (za * ze).sum(1, keepdim=True) / a.tau], dim=1)
        loss = F.cross_entropy(logits, torch.arange(a.batch, device=dev)) + a.beta * model.logdet_reg()
        opt.zero_grad(); loss.backward(); opt.step()
        ema = loss.item() if ema is None else 0.99 * ema + 0.01 * loss.item()
        if step % 500 == 0 or step == a.steps - 1:
            print(f"  step {step:5d} ema {ema:.4f} [{time.time() - t0:.0f}s]", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({"state": model.state_dict(), "L": L, "D": D, "n_params": int(n_params),
                "n_triplets": int(len(T)), "cutoff": int(a.cutoff), "steps": int(a.steps)}, a.out)
    with open(a.out + ".json", "w") as fh:
        json.dump({"L": L, "D": D, "n_params": int(n_params), "n_triplets": int(len(T)),
                   "cutoff": int(a.cutoff), "steps": int(a.steps), "final_ema_loss": ema}, fh, indent=1)
    print(f"[saved] {a.out}  ({n_params:,} params, {len(T):,} triplets, "
          f"{time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
