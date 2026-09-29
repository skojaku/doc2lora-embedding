"""M3 (nearest-category forced choice) for the four M-C label sets (#152).

Reuses `label_eval_metric3.py`'s option construction and judge prompt verbatim, so the
scores are comparable with the existing cluster-label table, with one deliberate
change: the panel is the judge roster MINUS the namer, not the full `JUDGES` that
`label_eval_metric3.py` imports.

That change is required, not cosmetic. `minimax/minimax-m3` names the continuation text
in `arith_name.py`, and it sits *inside* the legacy roster — judging with it would let
the namer grade its own output on the `*_cont_named` arms. `PANEL` excludes minimax,
so namer and panel are disjoint (#146).

The `_control_gt` source is kept: the node's true label is judged too, and the panel
should pick "ground_truth" at ~1.0. If it does not, the choice task is broken and no
other row means anything.

  set -a; . ../../.env; set +a
  python arith_m3.py
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from bench_data import out_dir   # bench_data.py sits next to this file
DATA = out_dir("actpatch")           # where this chain writes

ROOT = HERE.parents[1]
BT = ROOT / "data/labels"
sys.path.insert(0, str(BT))
from label_eval_judges import JUDGES, judge_json, run_jobs  # noqa: E402
from label_eval_metric3 import LETTERS, SYS, options_for, build_user  # noqa: E402

ARMS = ["gene_label", "act_label", "gene_cont_named", "act_cont_named"]
BOOT = 2000

# `minimax/minimax-m3` names the continuation text in arith_name.py AND sits in the
# judge roster, so judging with the full roster would let the namer grade its own
# output on the *_cont_named arms. Drop it from the panel here; the remaining four
# labs judge. Disjoint namer and panel, per #146.
NAMER_SLUG = "minimax/minimax-m3"
PANEL = {k: v for k, v in JUDGES.items() if v != NAMER_SLUG}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA / "arith_m3.json"))
    a = ap.parse_args()

    nodes = json.loads((BT / "label_eval_nodes.json").read_text())
    rows = json.loads((DATA / "arith_m1.json").read_text())["rows"]
    opts_by_code = {n["code"]: options_for(n) for n in nodes}

    sources = ["_control_gt"] + ARMS
    jobs = []
    for n in nodes:
        if n["code"] not in rows:
            continue
        for src in sources:
            dec = n["gt_label"] if src == "_control_gt" else rows[n["code"]].get(src)
            for jname, slug in PANEL.items():
                jobs.append({"code": n["code"], "src": src, "judge": jname,
                             "slug": slug, "decoded": dec})
    print(f"M3: {len(opts_by_code)} nodes x {len(sources)} sources x "
          f"{len(PANEL)} judges = {len(jobs)} calls (cached)", flush=True)

    def run(job):
        if not job["decoded"]:
            return {**job, "role": None}
        opts = opts_by_code[job["code"]]
        v = judge_json(job["slug"], SYS, build_user(job["decoded"], opts))
        role = None
        if "_error" not in v:
            i = LETTERS.find(str(v.get("choice", "")).strip().upper()[:1])
            role = opts[i][0] if 0 <= i < len(opts) else None
        return {**job, "role": role}

    res = run_jobs(jobs, run, max_workers=8, every=60)

    rng = np.random.default_rng(0)
    out = {"panel": list(PANEL), "namer": NAMER_SLUG,
           "namer_excluded_from_panel": True, "arms": {}}
    for src in sources:
        per_node, roledist, per_judge = [], {}, {}
        for code in opts_by_code:
            v = [r["role"] for r in res if r["code"] == code and r["src"] == src]
            v = [x for x in v if x is not None]
            if v:
                per_node.append(sum(x == "ground_truth" for x in v) / len(v))
            for x in v:
                roledist[x] = roledist.get(x, 0) + 1
        for j in PANEL:
            v = [r["role"] for r in res if r["src"] == src and r["judge"] == j]
            v = [x for x in v if x is not None]
            per_judge[j] = round(sum(x == "ground_truth" for x in v) / len(v), 3) if v else None
        if per_node:
            arr = np.array(per_node)
            sd = float(arr[rng.integers(0, len(arr), (BOOT, len(arr)))].mean(1).std())
            out["arms"][src] = {"m3": float(arr.mean()), "m3_sd": sd,
                                "n_nodes": len(arr), "role_dist": roledist,
                                "per_judge": per_judge}
            tot = sum(roledist.values()) or 1
            print(f"  {src:<16} M3 {arr.mean():.3f} +- {sd:.3f}   "
                  + "  ".join(f"{k}={roledist.get(k,0)/tot:.2f}"
                              for k in ("ground_truth", "parent", "child", "other")),
                  flush=True)

    # paired gene-vs-act differences on the same nodes
    def vec(src):
        d = {}
        for code in opts_by_code:
            v = [r["role"] for r in res if r["code"] == code and r["src"] == src]
            v = [x for x in v if x is not None]
            if v:
                d[code] = sum(x == "ground_truth" for x in v) / len(v)
        return d

    out["paired"] = {}
    for x, y in (("gene_label", "act_label"), ("gene_cont_named", "act_cont_named")):
        dx, dy = vec(x), vec(y)
        common = sorted(set(dx) & set(dy))
        d0 = np.array([dx[c] - dy[c] for c in common])
        bs = d0[rng.integers(0, len(d0), (5000, len(d0)))].mean(1)
        out["paired"][f"{x}-{y}"] = {
            "diff": float(d0.mean()), "sd": float(bs.std()),
            "p_le_0": float((bs <= 0).mean()), "n": len(common),
            "win": int((d0 > 0).sum()), "loss": int((d0 < 0).sum()),
            "tie": int((d0 == 0).sum())}
        p = out["paired"][f"{x}-{y}"]
        print(f"  [paired] {x} - {y}: {p['diff']:+.3f} +- {p['sd']:.3f}  "
              f"P(<=0)={p['p_le_0']:.3f}  win/loss/tie {p['win']}/{p['loss']}/{p['tie']}")

    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
