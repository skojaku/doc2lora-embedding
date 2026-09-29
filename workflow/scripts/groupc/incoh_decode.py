"""[GPU] Decode the #101 cluster centroids with the verbatim 2-3 word field prompt.

Three decode conditions, all with the same prompt and the same greedy settings:
  real            -- the PACS node centroid                       (the manuscript's condition)
  control_native  -- the cross-chapter centroid as it comes out   (norm NOT matched)
  control_matched -- the cross-chapter centroid rescaled to the matched real node's norm
                     (the alpha-scaling operation of Sec. 4.3: same norm, no shared topic)

Out: {condition: {cluster_id: {label, norm, raw}}} + the norms, so the norm-matching is auditable.

Usage: NEED_MB=22000 bash workflow/scripts/gpu_lease.sh \
         python workflow/scripts/groupc/incoh_decode.py --real ... --control ... --out ...
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d2l  # noqa: E402


def load_means(paths):
    out = {}
    for p in paths:
        z = np.load(p, allow_pickle=True)
        for i, k in enumerate(z["ids"]):
            out[str(k)] = z["means"][i].astype(np.float32)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--real", nargs="+", required=True, help="real-arm means npz shards")
    p.add_argument("--control", nargs="+", required=True, help="control-arm means npz shards")
    p.add_argument("--clusters", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--max_new_tokens", type=int, default=16)
    a = p.parse_args()

    real = load_means(a.real)
    ctrl = load_means(a.control)
    spec = json.load(open(a.clusters))
    match = {c["cid"]: c["match_node"] for c in spec["control"]}
    print(f"[incoh-decode] {len(real)} real, {len(ctrl)} control centroids", flush=True)

    model, gen_tok, ctx_tok = d2l.load(a.ckpt, mode="full")
    out = {"real": {}, "control_native": {}, "control_matched": {}, "norms": {}}

    for k, m in sorted(real.items()):
        raw = d2l.decode(model, gen_tok, m, d2l.FIELD23_PROMPT, a.max_new_tokens)
        out["real"][k] = {"label": d2l.normalize_label(raw), "raw": raw,
                          "norm": float(np.linalg.norm(m))}
        print(f"  real   [{k}] {out['real'][k]['label']}", flush=True)

    for k, m in sorted(ctrl.items()):
        node = match.get(k, "")
        n_native = float(np.linalg.norm(m))
        n_target = float(np.linalg.norm(real[node])) if node in real else n_native
        raw = d2l.decode(model, gen_tok, m, d2l.FIELD23_PROMPT, a.max_new_tokens)
        out["control_native"][k] = {"label": d2l.normalize_label(raw), "raw": raw,
                                    "norm": n_native, "match_node": node}
        scaled = m * (n_target / max(n_native, 1e-9))
        raw2 = d2l.decode(model, gen_tok, scaled, d2l.FIELD23_PROMPT, a.max_new_tokens)
        out["control_matched"][k] = {"label": d2l.normalize_label(raw2), "raw": raw2,
                                     "norm": float(np.linalg.norm(scaled)), "match_node": node,
                                     "scale": n_target / max(n_native, 1e-9)}
        out["norms"][k] = {"native": n_native, "target": n_target}
        print(f"  ctrl   [{k}] native={out['control_native'][k]['label']!r} "
              f"matched={out['control_matched'][k]['label']!r} "
              f"(|m| {n_native:.3f} -> {n_target:.3f})", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
