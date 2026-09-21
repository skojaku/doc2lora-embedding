"""[GPU] Train the invertible per-token ICAE adapter on OpenAlex citation triplets, mirroring the
doc2lora general adapter (train_general.py): SPECTER2-style InfoNCE with explicit hard+easy negatives.

The adapter is kron.KronAdapter(L=128, d=4096): per memory token a COMMON invertible map
C = Q diag(exp(s)) (shared rotation + dimensional scaling) plus a PER-TOKEN scalar a_l = exp(alpha_l).
The benchmark/loss embedding = mean over the 128 adapted slots -> 4096-dim. Because C is orthogonal-
times-positive-diagonal and a_l>0, the slot map is exactly invertible, so the compressed memory stays
decodable through the frozen ICAE decoder (decode_roundtrip.py).

Usage: CUDA_VISIBLE_DEVICES=0 STEPS=8000 python train_icae_adapter.py
Output: adapter_icae.pt  ({state, L=128, D=4096})
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "exps/2026-06-09-kron-adapter")
from kron import KronAdapter

GA = "exps/2026-06-10-general-adapter"
STEPS = int(os.environ.get("STEPS", "8000"))
BETA = float(os.environ.get("BETA", "0"))     # no logdet reg (broad data + hard negatives), as in train_general
TAU = 0.05
BS = int(os.environ.get("BS", "128"))         # smaller than genes' 256: slots are 128x heavier
dev = "cuda"

T = pd.read_parquet(f"{GA}/triplets.parquet")
pids = np.load(f"{HERE}/pool_pids.npy")
gk = {int(p): i for i, p in enumerate(pids)}
print("[icae] loading pool slots into RAM ...", flush=True)
slots = np.load(f"{HERE}/pool_slots.npy")                       # full 175GB into RAM (995GB box) -> fast gathers
N, MEM, D = slots.shape
L = MEM                                                        # one "token" per memory slot
ok = T.a.isin(gk) & T.pos.isin(gk) & T.neg_easy.isin(gk) & T.neg_hard.isin(gk)
T = T[ok].reset_index(drop=True)
ia, ip = T.a.map(gk).values, T.pos.map(gk).values
ie, ih = T.neg_easy.map(gk).values, T.neg_hard.map(gk).values
print(f"[icae] {len(T):,} triplets, slots {slots.shape}, L={L} d={D}, steps={STEPS} bs={BS}", flush=True)

model = KronAdapter(L=L, d=D).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
rng = np.random.default_rng(0); t0 = time.time(); ema = None


def emb(idx):
    # gather slots for a batch from the memmap, adapt per-token, mean-pool tokens -> [b, D]
    x = torch.from_numpy(np.ascontiguousarray(slots[idx])).to(dev).float()   # [b,128,4096]
    z = model(x).view(len(idx), L, D).mean(1)                                # pool the 128 adapted slots
    return z


for step in range(STEPS):
    sel = rng.integers(0, len(T), BS)
    za = F.normalize(emb(ia[sel]), dim=1)
    zp = F.normalize(emb(ip[sel]), dim=1)
    zh = F.normalize(emb(ih[sel]), dim=1)
    ze = F.normalize(emb(ie[sel]), dim=1)
    s_pp = za @ zp.t() / TAU
    s_h = (za * zh).sum(1, keepdim=True) / TAU
    s_e = (za * ze).sum(1, keepdim=True) / TAU
    logits = torch.cat([s_pp, s_h, s_e], dim=1)
    labels = torch.arange(BS, device=dev)
    loss = F.cross_entropy(logits, labels) + BETA * model.logdet_reg()
    opt.zero_grad(); loss.backward(); opt.step()
    ema = loss.item() if ema is None else 0.99 * ema + 0.01 * loss.item()
    if step % 200 == 0 or step == STEPS - 1:
        print(f"  step {step:5d} ema {ema:.4f} [{time.time()-t0:.0f}s]", flush=True)

torch.save({"state": model.state_dict(), "L": L, "D": D}, f"{HERE}/adapter_icae.pt")
print(f"[saved] {HERE}/adapter_icae.pt ({time.time()-t0:.0f}s)", flush=True)
