"""Metric 3 (nearest-category classification) for cluster-labeling (issue #24).

The plain match/partial/wrong judge saturated. Instead, this is a discriminative
forced choice: the judge is shown a decoded label and must place it on the single
CLOSEST category among the node's own ground-truth label, its PARENT (broader), a
CHILD (narrower), or "other (a clearly different field)". Scoring "picked the
ground-truth" turns over-broadening into a parent vote, over-narrowing into a
child vote, and off-topic into an other vote -- so the score spreads out.

Parent/child are the ancestor/descendant sampled in label_eval_prep.py; options
are shuffled per node (seeded) to avoid position bias. Judged by the six-model
roster, with across-judge spread.

CONTROL: the node's true label is also classified (source "_control_gt"); the
judge should pick "ground_truth" (~1.0), confirming the choice task works.

Reads:  label_eval_nodes.json
Writes: label_eval_metric3.json
Run:    set -a; . ../../.env; set +a; python label_eval_metric3.py
"""
import hashlib
import json
import random
import statistics as st
from pathlib import Path

from label_eval_judges import JUDGES, judge_json, run_jobs

HERE = Path(__file__).resolve().parent
METHODS = ["doc2lora", "vec2text", "keyllm", "icae", "incontext"]
LETTERS = "ABCDEF"

SYS = ("You assign a candidate topic label to the single closest category from a "
       "fixed list. Judge by meaning. Reply with ONLY a JSON object.")


def options_for(node):
    """[(role, label)] present for this node, plus the always-present 'other'."""
    opts = [("ground_truth", node["gt_label"])]
    if node.get("ancestor"):
        opts.append(("parent", node["ancestor"]["label"]))
    if node.get("descendant"):
        opts.append(("child", node["descendant"]["label"]))
    # Seed from a STABLE digest of the code, not hash(): Python randomises string
    # hashing per process, so hash() reshuffled the options on every run, missed the
    # response cache every time, and made the metric irreproducible run-to-run.
    seed = int(hashlib.sha1(node["code"].encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    rng.shuffle(opts)
    opts.append(("other", "None of the above (a clearly different field)"))
    return opts


def build_user(decoded, opts):
    lines = [f'A topic label was produced for a cluster of physics papers:\n  "{decoded}"',
             "", "Which ONE category does this label most closely describe?"]
    for L, (_role, lab) in zip(LETTERS, opts):
        lines.append(f"  {L}. {lab}")
    lines += ["",
              "Pick exactly one letter (the closest single category).",
              'Return ONLY: {"choice": "' + "/".join(LETTERS[:len(opts)])
              + '", "reason": "<=15 words"}']
    return "\n".join(lines)


def main():
    nodes = json.loads((HERE / "label_eval_nodes.json").read_text())
    opts_by_code = {n["code"]: options_for(n) for n in nodes}

    sources = ["_control_gt"] + METHODS
    jobs = []
    for n in nodes:
        for src in sources:
            decoded = n["gt_label"] if src == "_control_gt" else n["decoded"].get(src)
            for jname, slug in JUDGES.items():
                jobs.append({"code": n["code"], "src": src, "judge": jname,
                             "slug": slug, "decoded": decoded})

    print(f"Metric 3: {len(nodes)} nodes x {len(sources)} sources x {len(JUDGES)} judges "
          f"= {len(jobs)} calls (cached)")

    def run(job):
        if not job["decoded"]:
            return {**job, "verdict": {"_error": "no label"}}
        opts = opts_by_code[job["code"]]
        v = judge_json(job["slug"], SYS, build_user(job["decoded"], opts))
        role = None
        if "_error" not in v:
            letter = str(v.get("choice", "")).strip().upper()[:1]
            i = LETTERS.find(letter)
            role = opts[i][0] if 0 <= i < len(opts) else None
        return {**job, "verdict": {**v, "choice_role": role}}

    res = run_jobs(jobs, run, max_workers=8, every=60)

    # (code, src, judge) -> 1 if picked ground_truth else 0 (None on error)
    correct = {}
    roledist = {m: {"ground_truth": 0, "parent": 0, "child": 0, "other": 0, "na": 0}
                for m in METHODS}
    for r in res:
        role = r["verdict"].get("choice_role")
        ok = None if role is None else int(role == "ground_truth")
        correct[(r["code"], r["src"], r["judge"])] = ok
        if r["src"] in METHODS:
            roledist[r["src"]][role if role in roledist[r["src"]] else "na"] += 1

    def per_judge_acc(src):
        out = {}
        for jn in JUDGES:
            vals = [correct[(n["code"], src, jn)] for n in nodes]
            vals = [x for x in vals if x is not None]
            out[jn] = round(sum(vals) / len(vals), 4) if vals else None
        return out

    sane = per_judge_acc("_control_gt")
    sane_vals = [v for v in sane.values() if v is not None]
    summary = {"sanity_gt_vs_gt": {"per_judge": sane,
                                   "mean": round(sum(sane_vals) / len(sane_vals), 4)}}
    methods_out = {}
    for m in METHODS:
        pj = per_judge_acc(m)
        vals = [v for v in pj.values() if v is not None]
        methods_out[m] = {"per_judge": pj, "mean": round(sum(vals) / len(vals), 4),
                          "spread_sd": round(st.pstdev(vals), 4) if len(vals) > 1 else 0.0,
                          "choice_distribution": roledist[m]}
    summary["methods"] = methods_out

    detail = [{"code": r["code"], "src": r["src"], "judge": r["judge"],
               "decoded": r["decoded"], "verdict": r["verdict"]} for r in res]
    out = {"metric": "nearest_category_classification", "n_nodes": len(nodes),
           "summary": summary, "detail": detail}
    (HERE / "label_eval_metric3.json").write_text(json.dumps(out, indent=2))

    print(f"\nControl (classify the TRUE label; should pick ground_truth): "
          f"mean={summary['sanity_gt_vs_gt']['mean']}")
    print("\nMetric 3 — nearest-category accuracy (picked ground-truth; across judges):")
    for m in METHODS:
        s = methods_out[m]
        d = s["choice_distribution"]
        print(f"  {m:10s} {s['mean']:.3f}  (±{s['spread_sd']:.3f})  "
              f"[gt {d['ground_truth']} / parent {d['parent']} / child {d['child']} / other {d['other']}]")
    print(f"\nwrote {HERE / 'label_eval_metric3.json'}")


if __name__ == "__main__":
    main()
