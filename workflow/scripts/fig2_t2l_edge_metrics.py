"""Fig. 2 (e), (f), the T2L arm: SBERT position and verbatim copy rate of the
Text-to-LoRA decodes along the A-B edge (#151).

T2L was decoded on the same 13-point alpha grid as the other arms (0 to 1 in
steps of 1/12, weight on B) by workflow/scripts/t2l_fusion.py
(t2l.smk:t2l_fusion), in each of its three candidate embedding spaces (e0 = frozen gte output, e1 = TaskEncoder
output, e2 = generated LoRA factors). This script scores one space with the
same two statistics the other arms carry (pair_axis_metrics.py /
pair_axis_copyrate.py):

  mix_t   dot(d - A, B - A) / |B - A|^2 in SBERT (all-mpnet-base-v2) space
  copy    fraction of the decode's word tokens found verbatim in either lead

and writes a pair_axis-shaped cache: {set: {"stratum", "pts": {"mix": {alpha:
t}, "copy": {alpha: rate}}}}. CPU, seconds.

    python workflow/scripts/fig2_t2l_edge_metrics.py \
        --midpoints data/t2l/results/midpoints_t2l_e1.json \
        --out data/pair_axis/results/pair_axis_t2l_e1_pairaxis.json
"""
import argparse
import json
import os
import re
import sys

import numpy as np


def toks(t):
    return re.findall(r"[a-z][a-z-]{2,}", (t or "").lower())


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def main(midpoints, out, device="cpu"):
    D = json.load(open(midpoints))
    from sentence_transformers import SentenceTransformer
    sb = SentenceTransformer("all-mpnet-base-v2", device=device)

    def emb(t):
        return unit(np.atleast_2d(sb.encode(t, normalize_embeddings=True, show_progress_bar=False)).ravel())

    res = {}
    for p in D["pairs"]:
        eA, eB = emb(p["A"]), emb(p["B"])
        axis = eB - eA
        denom = float(axis @ axis) + 1e-12
        srctok = set(toks(p["A"])) | set(toks(p["B"]))
        mix, copy = {}, {}
        for al, text in p["decodes"].items():
            key = str(round(float(al), 4))
            at = toks(text)
            if len(at) < 6:
                continue
            mix[key] = float((emb(text) - eA) @ axis / denom)
            copy[key] = sum(w in srctok for w in at) / len(at)
        res[p["set"]] = {"stratum": p["stratum"], "pts": {"mix": mix, "copy": copy}}
    meta = {"_arm": D.get("arm"), "_source": os.path.basename(midpoints), "_alphas": D.get("alphas"),
            "_sbert": "all-mpnet-base-v2"}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump({**meta, **res}, open(out, "w"), indent=1)
    print(f"wrote {out} ({len(res)} pairs)")


if "snakemake" in sys.modules:
    main(snakemake.input["midpoints"], snakemake.output["cache"], snakemake.params["device"])
elif __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--midpoints", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    main(a.midpoints, a.out, a.device)
