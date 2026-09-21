"""[GPU] ICAE-embed a benchmark subset and write BOTH the raw ICAE baseline and the adapter-applied
variant, pooled to 4096-dim. Slots are computed in-memory and pooled immediately, so benchmark papers
never need 128-slot disk storage (only the training pool does).

  icae           = mean over the 128 memory slots                      -> [N,4096]  (raw baseline)
  icae_genkron   = mean over the 128 ADAPTED slots (invertible per-token map) -> [N,4096]

kinds:
  field <economics|psychology>  -> data/fields/<f>/embeddings/icae{,_genkron}_emb.npz  (subset from collector)
  aps                           -> data/aps/embeddings/icae{,_genkron}_emb.npz          (subset from collector)
  s2and <ds>                    -> exps/2026-06-09-s2and/proc/<ds>/icae{,_genkron}.npz   (all papers)

Usage: CUDA_VISIBLE_DEVICES=0 BATCH=24 python embed_apply.py field economics
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "exps/2026-06-09-kron-adapter")
from icae_lib import load_icae, compress_batch
from kron import KronAdapter

W = "exps/2026-05-26-baseline/icae/weights/mistral_7b_ft_icae.safetensors"
CODE = "exps/2026-05-26-baseline/icae/code/icae_v2"
BASE = "mistralai/Mistral-7B-Instruct-v0.2"
BATCH = int(os.environ.get("BATCH", "24"))
S2 = "exps/2026-06-09-s2and"
dev = "cuda"


def resolve(kind, name):
    """Return (ids[list], texts[list], out_dir, id_is_str). Subset ids come from the collector for
    field/aps; for s2and we embed every paper in the dataset."""
    if kind == "field":
        ids = pd.read_parquet(f"{HERE}/{name}_eval_ids.parquet").paper_id.astype(int).tolist()
        tx = pd.read_parquet(f"data/fields/{name}/paper_text.parquet")
        out = f"data/fields/{name}/embeddings"; key = "paper_id"; sstr = False
    elif kind == "aps":
        ids = pd.read_parquet(f"{HERE}/aps_eval_ids.parquet").paper_id.astype(int).tolist()
        tx = pd.read_parquet("data/aps/paper_text.parquet").rename(columns={"aps_paper_id": "paper_id"})
        out = "data/aps/embeddings"; key = "paper_id"; sstr = False
    elif kind == "s2and":
        z = np.load(f"{S2}/proc/{name}/specter.npz", allow_pickle=True)
        ids = [str(p) for p in z["paper_ids"]]
        tx = pd.read_parquet(f"{S2}/proc/{name}/paper_text.parquet"); tx["paper_id"] = tx.paper_id.astype(str)
        out = f"{S2}/proc/{name}"; key = "paper_id"; sstr = True
    else:
        raise SystemExit(f"unknown kind {kind}")
    txt = dict(zip(tx[key].astype(str) if sstr else tx[key].astype(int), tx.text))
    keep = [i for i in ids if i in txt]
    texts = [txt[i] if isinstance(txt[i], str) and txt[i].strip() else "." for i in keep]
    return keep, texts, out, sstr


def main(kind, name):
    ids, texts, out, sstr = resolve(kind, name)
    N = len(ids)
    print(f"[{kind}:{name}] embedding {N:,} papers (batch={BATCH})", flush=True)
    model = load_icae(W, CODE, BASE, dev)
    D = model.dim

    ck = torch.load(f"{HERE}/adapter_icae.pt", map_location=dev)
    A = KronAdapter(L=ck["L"], d=ck["D"]).to(dev); A.load_state_dict(ck["state"]); A.eval()
    with torch.no_grad():
        C = A.C()                                          # cache shared rotation+scale once
        alpha = torch.exp(A.alpha)[None, :, None]          # [1,128,1] per-token scale
    raw = np.empty((N, D), np.float16); gen = np.empty((N, D), np.float16)

    # length-bucketing: process short docs together so left-padding waste (and wasted FLOPs) is minimal.
    order = sorted(range(N), key=lambda k: len(texts[k]))            # ascending char length ~ token length
    t0 = time.time()
    with torch.no_grad():
        for bi in range(0, N, BATCH):
            idx = order[bi:bi + BATCH]
            s = compress_batch(model, [texts[k] for k in idx], dev)  # [b,128,4096] f32
            rmean = s.mean(1).half().cpu().numpy()
            z = (torch.einsum("bld,de->ble", s, C) * alpha).mean(1).half().cpu().numpy()  # adapted+pooled
            for j, k in enumerate(idx):
                raw[k] = rmean[j]; gen[k] = z[j]
            if bi % (BATCH * 100) == 0:
                el = time.time() - t0
                print(f"  {min(bi+BATCH,N):,}/{N:,}  {el:.0f}s  {(bi+BATCH)/max(el,1):.0f} doc/s", flush=True)

    pid_arr = np.array(ids) if sstr else np.array(ids, dtype=np.int64)
    np.savez(f"{out}/icae.npz" if kind == "s2and" else f"{out}/icae_emb.npz",
             vecs=raw, paper_ids=pid_arr)
    np.savez(f"{out}/icae_genkron.npz" if kind == "s2and" else f"{out}/icae_genkron_emb.npz",
             vecs=gen, paper_ids=pid_arr)
    print(f"[saved] {out} icae + icae_genkron  ({N} x {D}, {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
