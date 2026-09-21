"""[GPU] Decode every point under K paraphrases of its prompt (#73).

Three decode families, so the answer covers each kind of text the manuscript reports:
  label    the PACS node centroids, 2-3 word field prompt         (Sec. 4.2 labels)
  describe single papers from the #95 sample                      (document points)
  fusion   midpoints of paper pairs from the #95 sample           (Sec. 4.4 composition)

Usage: NEED_MB=20000 bash workflow/scripts/gpu_lease.sh \
         python workflow/scripts/groupc/psens_decode.py --family label --out ...
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d2l  # noqa: E402
from prompts import DESCRIBE_PROMPTS, FUSION_PROMPTS, LABEL_PROMPTS  # noqa: E402

FAMILIES = {"label": (LABEL_PROMPTS, 16), "describe": (DESCRIBE_PROMPTS, 160),
            "fusion": (FUSION_PROMPTS, 120)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=list(FAMILIES), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--means", default="exps/2026-06-11-baseline-trees/qwen_fullrank_means.npz")
    p.add_argument("--sample", default="data/groupc/fidelity/sample.parquet")
    p.add_argument("--n_units", type=int, default=40)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    prompts, maxnew = FAMILIES[a.family]
    prompts = prompts[:a.k]
    model, gen_tok, ctx_tok = d2l.load(a.ckpt, mode="full")

    units = []          # (unit_id, tensor)
    if a.family == "label":
        z = np.load(a.means, allow_pickle=True)
        for k, m in zip([str(x) for x in z["nodes"]], z["means"].astype(np.float32)):
            units.append((k, m))
    else:
        df = pd.read_parquet(a.sample)
        rng = np.random.default_rng(a.seed)
        idx = rng.permutation(len(df))
        texts = df.text.astype(str).tolist()
        pids = df.paper_id.astype(np.int64).tolist()
        if a.family == "describe":
            for i in idx[:a.n_units]:
                emb = d2l.extract_full(model, ctx_tok, [texts[i]])[0]
                units.append((str(pids[i]), emb.detach().cpu().numpy().squeeze(1)
                              if emb.dim() == 4 else emb.detach().cpu().numpy()))
        else:   # fusion: midpoint of disjoint pairs
            pairs = [(int(idx[2 * t]), int(idx[2 * t + 1])) for t in range(a.n_units)
                     if 2 * t + 1 < len(idx)]
            for i, jx in pairs:
                e = d2l.extract_full(model, ctx_tok, [texts[i], texts[jx]])
                A = e[0].detach().cpu().numpy(); B = e[1].detach().cpu().numpy()
                A = A.squeeze(1) if A.ndim == 4 else A
                B = B.squeeze(1) if B.ndim == 4 else B
                units.append((f"{pids[i]}+{pids[jx]}", 0.5 * (A + B)))
    print(f"[psens {a.family}] {len(units)} units x {len(prompts)} prompts", flush=True)

    out = {}
    for uid, T in units:
        rec = []
        for pi, pr in enumerate(prompts):
            raw = d2l.decode(model, gen_tok, T, pr, max_new_tokens=maxnew)
            txt = d2l.normalize_label(raw) if a.family == "label" else raw.strip()
            rec.append({"prompt_idx": pi, "text": txt})
        out[uid] = rec
        print(f"  [{uid}] " + " | ".join(r["text"][:40] for r in rec[:4]), flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"family": a.family, "prompts": prompts, "units": out}, fh,
                  ensure_ascii=False, indent=1)
    print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
