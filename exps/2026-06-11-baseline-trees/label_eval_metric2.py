"""Metric 2 (hierarchical consistency) for cluster-labeling (issue #24).

Tests whether a decoded label respects the PACS tree: it should be NARROWER than
a sampled ancestor and BROADER than a sampled descendant. Judged by the six-model
roster (label_eval_judges.JUDGES), reporting across-judge spread.

VALIDATION GATE (run first): the same test is applied to the *true* PACS label of
each node. The judges must confirm the true label is narrower-than-ancestor (and,
where a descendant exists, broader-than-descendant). Method scores are only
reported if the gate passes (mean confirmation over nodes x judges >= GATE_THRESH).

Reads:  label_eval_nodes.json
Writes: label_eval_metric2.json
Run:    set -a; . ../../.env; set +a; python label_eval_metric2.py
"""
import json
import statistics as st
from pathlib import Path

from label_eval_judges import JUDGES, judge_json, run_jobs

HERE = Path(__file__).resolve().parent
METHODS = ["doc2lora", "vec2text", "keyllm", "icae", "incontext"]
GATE_THRESH = 0.75

SYS = ("You judge whether a candidate label for a topic cluster sits at the correct "
       "level of generality inside a known concept hierarchy. Judge by meaning, not "
       "wording. Reply with ONLY a JSON object.")


def build_user(focal, ancestor, descendant):
    lines = [
        "Concept hierarchy context:",
        f'- BROADER parent concept: "{ancestor}"',
    ]
    if descendant:
        lines.append(f'- NARROWER child concept: "{descendant}"')
    lines += [
        "",
        f'Candidate label for the focal node (which sits below the parent'
        f'{" and above the child" if descendant else ""}): "{focal}"',
        "",
        f'1) Is the candidate MORE SPECIFIC (narrower) than the broader parent "{ancestor}"?',
    ]
    if descendant:
        lines.append(f'2) Is the candidate MORE GENERAL (broader) than the narrower child "{descendant}"?')
    lines += [
        "A label essentially the same scope as the parent is NOT narrower; one "
        "essentially the same scope as the child is NOT broader.",
        "",
        'Return ONLY: {"narrower_than_ancestor": true|false, '
        '"broader_than_descendant": ' + ("true|false" if descendant else "null")
        + ', "reason": "<=20 words"}',
    ]
    return "\n".join(lines)


def consistent(v, has_desc):
    if v.get("narrower_than_ancestor") is not True:
        return False
    return (v.get("broader_than_descendant") is True) if has_desc else True


def main():
    nodes = json.loads((HERE / "label_eval_nodes.json").read_text())
    nodes = [n for n in nodes if n.get("ancestor")]          # need at least an ancestor

    # focal sources: "_truth" (gate) + each method
    sources = ["_truth"] + METHODS
    jobs = []
    for n in nodes:
        anc = n["ancestor"]["label"]
        desc = n["descendant"]["label"] if n["descendant"] else None
        for src in sources:
            focal = n["gt_label"] if src == "_truth" else n["decoded"].get(src)
            for jname, slug in JUDGES.items():
                jobs.append({"code": n["code"], "src": src, "judge": jname,
                             "slug": slug, "focal": focal, "anc": anc, "desc": desc})

    print(f"Metric 2: {len(nodes)} nodes x {len(sources)} sources x {len(JUDGES)} judges "
          f"= {len(jobs)} calls (cached)")

    def run(job):
        if not job["focal"]:
            return {**job, "verdict": {"_error": "no label"}}
        v = judge_json(job["slug"], SYS, build_user(job["focal"], job["anc"], job["desc"]))
        return {**job, "verdict": v}

    res = run_jobs(jobs, run, max_workers=8, every=60)

    # index: (code, src, judge) -> consistent bool ; track has_desc per node
    has_desc = {n["code"]: bool(n["descendant"]) for n in nodes}
    flags = {}
    for r in res:
        v = r["verdict"]
        ok = consistent(v, has_desc[r["code"]]) if "_error" not in v else None
        flags[(r["code"], r["src"], r["judge"])] = ok

    def per_judge_rate(src):
        out = {}
        for jn in JUDGES:
            vals = [flags[(n["code"], src, jn)] for n in nodes]
            vals = [x for x in vals if x is not None]
            out[jn] = round(sum(vals) / len(vals), 4) if vals else None
        return out

    # gate
    gate_pj = per_judge_rate("_truth")
    gate_vals = [v for v in gate_pj.values() if v is not None]
    gate_mean = round(sum(gate_vals) / len(gate_vals), 4)
    gate_passed = gate_mean >= GATE_THRESH

    summary = {"gate": {"per_judge": gate_pj, "mean": gate_mean,
                        "threshold": GATE_THRESH, "passed": gate_passed}}

    if gate_passed:
        methods_out = {}
        for m in METHODS:
            pj = per_judge_rate(m)
            vals = [v for v in pj.values() if v is not None]
            methods_out[m] = {"per_judge": pj,
                              "mean": round(sum(vals) / len(vals), 4),
                              "spread_sd": round(st.pstdev(vals), 4) if len(vals) > 1 else 0.0}
        summary["methods"] = methods_out
    else:
        summary["methods"] = None
        summary["note"] = ("Validation gate FAILED: judges did not reliably confirm the "
                           "true PACS labels are hierarchically consistent. Method scores "
                           "withheld per the issue-#24 protocol.")

    detail = [{"code": r["code"], "src": r["src"], "judge": r["judge"],
               "focal": r["focal"], "verdict": r["verdict"]} for r in res]
    out = {"metric": "hierarchical_consistency", "n_nodes": len(nodes),
           "summary": summary, "detail": detail}
    (HERE / "label_eval_metric2.json").write_text(json.dumps(out, indent=2))

    print(f"\nValidation gate: mean={gate_mean} (thresh {GATE_THRESH}) -> "
          f"{'PASS' if gate_passed else 'FAIL'}")
    print("  per-judge:", gate_pj)
    if gate_passed:
        print("\nMetric 2 — hierarchical consistency (mean fraction over nodes, across judges):")
        for m in METHODS:
            s = summary["methods"][m]
            print(f"  {m:10s} {s['mean']:.3f}  (±{s['spread_sd']:.3f} across judges)")
    print(f"\nwrote {HERE / 'label_eval_metric2.json'}")


if __name__ == "__main__":
    main()
