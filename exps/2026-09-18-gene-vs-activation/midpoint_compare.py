"""Midpoints: does interpolation behave in either representation? (#152)

M-C averaged thousands of documents and found the gap between the gene and raw
activations nearly closes. This asks the other arithmetic question -- what happens
BETWEEN two documents -- on the 100 corner pairs already drawn for
`workflow/rules/simplex_kwgrid.smk`: 50 at PACS stratum L1 (near, same subtopic) and 50
at L5 (far, three chapters apart). Reusing that draw keeps this paired with §4.3.

Per pair we decode three points -- A (alpha 0), the midpoint (0.5), and B (1) -- in each
representation. The endpoints are not decoration: without them there is no way to tell a
midpoint that blends from a midpoint that simply snapped to one corner.

Metrics follow `workflow/scripts/fusion_vs_copy.py` so the numbers mean the same thing
as in the simplex work:

  min_sim       min over the two corners of SBERT cos(decode, corner lead)
  fusion_gain   min_sim - copy_floor, where for a two-corner cell copy_floor is just
                cos(A, B): the min-similarity a pure COPY of one corner already earns.
                Positive means the decode is closer to BOTH corners than the corners are
                to each other -- combination beyond copying.
  max_density   Grusky extractive density against either corner: the lexical copy
                detector.
  balance       min/max of the two corner similarities; 1 = both contribute equally.

The near/far split matters: at L1 the corners are already similar, so copy_floor is high
and fusion_gain is hard to earn; at L5 a real blend has room to show.

  CUDA_VISIBLE_DEVICES=0 python midpoint_compare.py --arm act
  PYTHONPATH=... DOC2LORA_CKPT=... CUDA_VISIBLE_DEVICES=1 python midpoint_compare.py --arm gene
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "workflow/scripts"))
from fusion_vs_copy import density, toks  # noqa: E402

KG = ROOT / "exps/2026-07-02-simplex-kwgrid"
BEST = dict(layer=15, slots=8, mode="mean")
ACT_CONT = "The following is a scientific abstract."
GENE_CONT = "The following is a scientific abstract. Write it."
ALPHAS = [0.0, 0.5, 1.0]
MAX_NEW = 64


def load_pairs(strata=("L1", "L5"), n=50):
    out = []
    for L in strata:
        for i in range(n):
            p = KG / f"corners_pair{L}_{i:02d}.json"
            if not p.exists():
                continue
            c = json.loads(p.read_text())["corners"]
            k = list(c)
            out.append({"set": f"{L}_{i:02d}", "stratum": L,
                        "A": c[k[0]]["lead"], "B": c[k[1]]["lead"],
                        "A_name": c[k[0]]["name"], "B_name": c[k[1]]["name"]})
    return out


def run_act(pairs):
    from act_common import load_qwen, extract_acts, _patch_at
    from layer_sweep import build

    Q = load_qwen()
    ids, pos = build(Q, BEST["slots"])          # ACT_CONT, raw completion context
    model, tok = Q["model"], Q["tok"]

    def dec(vec):
        vv = torch.as_tensor(vec).reshape(1, -1).repeat(len(pos), 1)
        with _patch_at(model, BEST["layer"], pos, vv), torch.no_grad():
            o = model.generate(input_ids=ids.to(Q["device"]), max_new_tokens=MAX_NEW,
                               do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip()

    out = []
    for j, p in enumerate(pairs):
        V = extract_acts(Q, [p["A"], p["B"]], BEST["layer"], mode=BEST["mode"])
        d = {a: dec((1 - a) * V[0] + a * V[1]) for a in ALPHAS}
        out.append({**p, "decodes": {str(a): d[a] for a in ALPHAS}})
        if j % 10 == 0:
            print(f"  [{j}/{len(pairs)}] mid={d[0.5][:90]!r}", flush=True)
    return out


def run_gene(pairs):
    import os

    sys.path.insert(0, str(ROOT / "libs/doc2lora"))
    from doc2lora import (load_model, extract_norm_lora_emb, interpolate_embeddings,
                          decode_adapter)

    ckpt = os.environ.get(
        "DOC2LORA_CKPT",
        str(ROOT / "data/agent_assets/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"))
    model, gen_tok, ctx_tok = load_model(ckpt, mode="full")

    out = []
    for j, p in enumerate(pairs):
        ea = extract_norm_lora_emb(model, ctx_tok, p["A"])
        eb = extract_norm_lora_emb(model, ctx_tok, p["B"])
        d = {}
        for a in ALPHAS:
            # `full` mode: the value-space interpolation that gave real fusion at the
            # midpoint in the arithmetic study (pooled mode is winner-take-all)
            emb = interpolate_embeddings(ea, eb, alpha=a, mode="full")
            d[a] = decode_adapter(model, gen_tok, emb, prompt=GENE_CONT,
                                  max_new_tokens=MAX_NEW)
        out.append({**p, "decodes": {str(a): d[a] for a in ALPHAS}})
        if j % 10 == 0:
            print(f"  [{j}/{len(pairs)}] mid={d[0.5][:90]!r}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["act", "gene"])
    ap.add_argument("--n", type=int, default=50, help="pairs per stratum")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or HERE / f"results/midpoints_{a.arm}.json")

    pairs = load_pairs(n=a.n)
    print(f"[pairs] {len(pairs)} "
          f"({sum(p['stratum']=='L1' for p in pairs)} L1 near / "
          f"{sum(p['stratum']=='L5' for p in pairs)} L5 far)", flush=True)

    res = run_act(pairs) if a.arm == "act" else run_gene(pairs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"arm": a.arm, "alphas": ALPHAS, "pairs": res}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
