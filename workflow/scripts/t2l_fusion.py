"""Text-to-LoRA fusion arm: midpoints between two documents (issue #151).

The prediction the issue sets up: averaging or interpolating in T2L's space and decoding
should NOT reproduce what §4.3 shows for \\doctolora, because T2L's coordinates belong to
a frozen gte encoder and nothing in its training reconstructs text. Either outcome is
publishable -- if T2L midpoints *do* blend, adapter-space smoothness is a general
property of hypernetwork adapters and §5's mechanism paragraph has to be rewritten.

Same 100 corner pairs as §4.3 and as the gene-vs-activation arm (#152): the existing
`data/pair_axis/corners_pair{L1,L5}_{00..49}.json`, 50 near + 50 far.
The pair axis is swept on the SAME alpha grid the other arms use -- 13 points, 0 to 1
in steps of 1/12 (pair_axis_metrics.py) -- so Fig. 2 (e), (f) can draw the T2L curve
next to the others instead of three isolated markers. Each alpha is interpolated in
each of the three embedding positions:

  E0  midpoint of the frozen gte vectors, then the whole hypernetwork runs on it
  E1  midpoint of the TaskEncoder outputs (64-d), then the rest of the hypernetwork
  E2  midpoint of the generated LoRA factors (A and B averaged separately, rank stays 8)

E2 is the structural analogue of what doc2lora does, since doc2lora's head is linear and
averaging its embeddings averages the emitted factors.

Scored by `workflow/scripts/midpoint_score.py`, so the numbers are
directly comparable with the gene and activation arms.

  export PYTHONPATH=$T2L_SRC
  CUDA_VISIBLE_DEVICES=0 python t2l_fusion.py
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
from bench_data import out_dir   # bench_data.py sits next to this file
DATA = out_dir("t2l")           # where this chain writes

ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from t2l_common import (  # noqa: E402
    load_t2l, embed_e0, e0_to_e1, e1_to_lora, mean_lora_sd, decode_lora,
)

KG = ROOT / "data/pair_axis"
# same instruction the gene arm receives, in T2L's own prompt format
CONT_PROMPT = "The following is a scientific abstract. Write it."
STEPS = 12                                             # 13 points: matches the alpha
ALPHAS = [round(i / STEPS, 4) for i in range(STEPS + 1)]   # keys of the other arms
MAX_NEW = 64
SPACES = ["e0", "e1", "e2"]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="pairs per stratum")
    ap.add_argument("--spaces", nargs="+", default=SPACES, choices=SPACES)
    ap.add_argument("--alphas", nargs="+", type=float, default=ALPHAS,
                    help="mixing weights on B; default = the 13-point grid of the other arms")
    ap.add_argument("--state", default=str(DATA / "t2l_fusion_state.json"))
    a = ap.parse_args()

    pairs = load_pairs(n=a.n)
    print(f"[pairs] {len(pairs)} "
          f"({sum(p['stratum']=='L1' for p in pairs)} L1 / "
          f"{sum(p['stratum']=='L5' for p in pairs)} L5)", flush=True)

    alphas = [round(float(x), 4) for x in a.alphas]
    state_path = Path(a.state)
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    # resume per (pair, space, alpha): a state written by an earlier, coarser grid keeps
    # its decodes and only the alphas missing from it are generated
    def missing(rec):
        if rec is None:
            return len(a.spaces) * len(alphas)
        return sum(str(al) not in rec.get("decodes", {}).get(sp, {})
                   for sp in a.spaces for al in alphas)

    todo = [p for p in pairs if missing(state.get(p["set"]))]
    n_dec = sum(missing(state.get(p["set"])) for p in pairs)
    print(f"[alphas] {len(alphas)}: {alphas}", flush=True)
    print(f"[todo] {len(todo)} pairs, {n_dec} decodes", flush=True)
    T = load_t2l() if todo else None

    for j, p in enumerate(todo):
        e0 = embed_e0(T, [p["A"], p["B"]])              # [2, 1024]
        e1 = e0_to_e1(T, e0)                            # [2, 64]
        sd_a, sd_b = e1_to_lora(T, e1[0]), e1_to_lora(T, e1[1])

        rec = state.get(p["set"]) or {"stratum": p["stratum"], "A": p["A"], "B": p["B"],
                                      "A_name": p["A_name"], "B_name": p["B_name"],
                                      "decodes": {}}
        for space in a.spaces:
            d = rec["decodes"].setdefault(space, {})
            for al in alphas:
                if str(al) in d:
                    continue
                if space == "e0":
                    v = (1 - al) * e0[0] + al * e0[1]
                    sd = e1_to_lora(T, e0_to_e1(T, v.unsqueeze(0))[0])
                elif space == "e1":
                    v = (1 - al) * e1[0] + al * e1[1]
                    sd = e1_to_lora(T, v)
                else:  # e2: factor-space mean, rank stays 8
                    sd = mean_lora_sd([sd_a, sd_b], weights=[1 - al, al])
                d[str(al)] = decode_lora(T, sd, CONT_PROMPT, max_new_tokens=MAX_NEW)
        state[p["set"]] = rec
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2))
        if j % 10 == 0:
            print(f"  [{j}/{len(todo)}] {p['set']} e2 mid="
                  f"{rec['decodes'].get('e2', {}).get('0.5', '')[:90]!r}", flush=True)

    # one file per space, in the shape midpoint_score.py reads
    for space in a.spaces:
        out = {"arm": f"t2l_{space}", "alphas": alphas,
               "pairs": [{"set": s, "stratum": v["stratum"], "A": v["A"], "B": v["B"],
                          "A_name": v["A_name"], "B_name": v["B_name"],
                          "decodes": v["decodes"][space]}
                         for s, v in state.items() if space in v["decodes"]]}
        p = DATA / f"midpoints_t2l_{space}.json"
        p.write_text(json.dumps(out, indent=2))
        print(f"wrote {p} ({len(out['pairs'])} pairs)")


if __name__ == "__main__":
    main()
