"""[GPU] Train the GENERAL doc2lora kron adapter on the fresh OpenAlex citation TRIPLETS
(anchor, positive, easy-neg, hard-neg) with SPECTER2-style InfoNCE: for each anchor the positive is
scored against all in-batch positives (easy negatives) PLUS its own explicit hard and easy negatives.

Usage: CUDA_VISIBLE_DEVICES=0 python train_general.py gemma
Output: data/general_adapter/adapter_general_<enc>.pt
"""
import sys, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, "data/kron")
from kron import KronAdapter

ENC = sys.argv[1] if len(sys.argv) > 1 else "gemma"
OUT = "data/general_adapter"
STEPS = int(os.environ.get("STEPS", "8000"))
BETA = float(os.environ.get("BETA", "0"))     # no regularization for the general adapter (broad data + hard negs)
TAU = 0.05
dev = "cuda"

T = pd.read_parquet(f"{OUT}/triplets.parquet")
z = np.load(f"{OUT}/pool_genes_{ENC}.npz", allow_pickle=True)
gk = {int(k): i for i, k in enumerate(z["pids"])}
G = z["embeddings"]; D = G.shape[1]; L = D // 512
# keep triplets whose 4 papers were all embedded
ok = T.a.isin(gk) & T.pos.isin(gk) & T.neg_easy.isin(gk) & T.neg_hard.isin(gk)
T = T[ok].reset_index(drop=True)
ia = T.a.map(gk).values; ip = T.pos.map(gk).values
ie = T.neg_easy.map(gk).values; ih = T.neg_hard.map(gk).values
print(f"[{ENC}] {len(T):,} triplets, pool {G.shape}, L={L}, steps={STEPS}", flush=True)

Gt = torch.from_numpy(G)                       # CPU fp16
model = KronAdapter(L=L, d=512).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
rng = np.random.default_rng(0); bs = 256; t0 = time.time(); ema = None


def emb(idx):
    return model(Gt[idx].view(len(idx), L, 512).float().to(dev))


for step in range(STEPS):
    sel = rng.integers(0, len(T), bs)
    za = F.normalize(emb(ia[sel]), dim=1)
    zp = F.normalize(emb(ip[sel]), dim=1)
    zh = F.normalize(emb(ih[sel]), dim=1)
    ze = F.normalize(emb(ie[sel]), dim=1)
    s_pp = za @ zp.t() / TAU                    # [B,B] in-batch positives + easy negatives
    s_h = (za * zh).sum(1, keepdim=True) / TAU  # [B,1] explicit hard negative
    s_e = (za * ze).sum(1, keepdim=True) / TAU  # [B,1] explicit easy negative
    logits = torch.cat([s_pp, s_h, s_e], dim=1)
    labels = torch.arange(bs, device=dev)
    loss = F.cross_entropy(logits, labels) + BETA * model.logdet_reg()
    opt.zero_grad(); loss.backward(); opt.step()
    ema = loss.item() if ema is None else 0.99 * ema + 0.01 * loss.item()
    if step % 500 == 0 or step == STEPS - 1:
        print(f"  step {step:5d} ema {ema:.4f} [{time.time()-t0:.0f}s]", flush=True)

torch.save({"state": model.state_dict(), "L": L, "D": D}, f"{OUT}/adapter_general_{ENC}.pt")
print(f"[saved] {OUT}/adapter_general_{ENC}.pt ({time.time()-t0:.0f}s)", flush=True)
