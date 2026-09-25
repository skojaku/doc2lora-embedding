"""Judge panel for the BERTopic arms (issue #24, metric 4 design).

label_eval_metric4.py exports its tournament as reusable pieces exactly so a
second experiment does not fork the design, so this imports SYS / build_user /
ask_order / collapse_orders verbatim and only changes the FIELD of arms. The
judge is shown the official PACS label plus two methods' labels for the same
node and picks the closer one, "tie" allowed; both orders are asked and a win
needs both to agree, so position bias is removed by construction.

Why a separate file rather than adding the arms to label_eval_metric4.py:
  - the win rate that script reports is scored against the field, so adding arms
    silently moves every incumbent's number. label_eval_metric4.json is tracked
    and Fig. 2 (b) reads it, so it is left exactly as it is.
  - the response cache is keyed on (model, system, the two labels + reference),
    so the pairs the paper already paid for are reused verbatim here and only
    the BERTopic pairs cost anything.

Field: the six methods Fig. 2 (b) shows (T2L as its TaskEncoder space,
t2l_hidden, as there), plus two BERTopic arms --
`bertopic` (c-TF-IDF top-10, BERTopic's default) and `bertopic3` (top-3, the
length-matched control). The KeyBERTInspired arms are deliberately NOT judged:
they were indistinguishable from plain c-TF-IDF on the semantic metric
(cosine 0.474 vs 0.477), so they would spend panel budget to re-measure a
null. Head-to-head rates are per-pair and stay comparable with the paper's;
the win-rate-against-the-field column is NOT comparable across fields.

The Vertex seat needs a PROJECT as well as credentials: _vertex_project() takes
GOOGLE_CLOUD_PROJECT, else the ADC file's quota_project_id. A
`gcloud auth application-default login` whose active gcloud configuration points
at a project without Vertex access writes the ADC file with no quota_project_id
(its set-quota-project step fails), and every gemini call then dies with
PERMISSION_DENIED after four backoff retries -- ~23 s per call, which looks like
slowness rather than failure. Export GOOGLE_CLOUD_PROJECT for the run, or fix it
once with `gcloud auth application-default set-quota-project <project>`.

Reads:  label_eval_nodes.json
Writes: bertopic_judge.json
Run:    python bertopic_judge.py            (needs OPENROUTER_API_KEY + ADC)
        GOOGLE_CLOUD_PROJECT=<vertex-enabled project> python bertopic_judge.py
        DRY=1 python bertopic_judge.py      (count uncached calls, spend nothing)
        JUDGES_SUBSET=glm-5.3-flash python bertopic_judge.py   (one seat)
        JUDGE_WORKERS=24 python bertopic_judge.py   (wider; the cache makes a
        restart free, so raising this only risks provider rate limits)
"""

import itertools
import json
import os
import statistics as st
from pathlib import Path

import numpy as np

from label_eval_judges import JUDGES_V2, _key, run_jobs          # noqa: E402
from label_eval_metric4 import (GT, SYS, ask_order, build_user,  # noqa: E402
                                collapse_orders)

HERE = Path(__file__).resolve().parent
from bench_data import out_dir   # bench_data.py sits next to this file
DATA = out_dir("labels")           # where this chain writes

OUT = DATA / "bertopic_judge.json"
METHODS = ["doc2lora", "icae", "incontext", "keyllm", "vec2text", "t2l_hidden",
           "bertopic", "bertopic3"]
NEW_ARMS = {"bertopic", "bertopic3"}
N_BOOT = 1000


def boot(xs, seed=0):
    x = np.asarray(xs, dtype=float)
    if not len(x):
        return 0.0
    rng = np.random.default_rng(seed)
    return float(x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(axis=1).std())


def main():
    nodes = json.loads((DATA / "label_eval_nodes.json").read_text())
    judges = dict(JUDGES_V2)
    if os.environ.get("JUDGES_SUBSET"):
        keep = set(os.environ["JUDGES_SUBSET"].split(","))
        judges = {k: v for k, v in judges.items() if k in keep}

    pairs = (list(itertools.combinations(METHODS, 2))
             + [(GT, m) for m in METHODS] + [(GT, GT)])

    def label_of(nd, m):
        return nd["gt_label"] if m == GT else nd["decoded"].get(m)

    jobs = []
    for nd in nodes:
        for x, y in pairs:
            lx, ly = label_of(nd, x), label_of(nd, y)
            if not lx or not ly:
                continue
            for order in (0, 1):
                jobs.append({"code": nd["code"], "gt": nd["gt_label"],
                             "x": x, "y": y, "lx": lx, "ly": ly, "order": order})
    jobs = [{**j, "judge": jn, "slug": sl} for j in jobs for jn, sl in judges.items()]

    def cached(job):
        a, b = (job["lx"], job["ly"]) if job["order"] == 0 else (job["ly"], job["lx"])
        return _key(job["slug"], SYS, build_user(job["gt"], a, b)).exists()

    miss = [j for j in jobs if not cached(j)]
    print(f"{len(nodes)} nodes x {len(pairs)} pairs x 2 orders x {len(judges)} judges "
          f"= {len(jobs)} calls, {len(jobs) - len(miss)} already cached, "
          f"{len(miss)} to spend", flush=True)
    if os.environ.get("DRY"):
        by_pair = {}
        for j in miss:
            by_pair["|".join(sorted((j["x"], j["y"])))] = by_pair.get(
                "|".join(sorted((j["x"], j["y"]))), 0) + 1
        for k, v in sorted(by_pair.items(), key=lambda kv: -kv[1]):
            print(f"  {v:6d}  {k}")
        return

    res = run_jobs(jobs, ask_order,
                   max_workers=int(os.environ.get("JUDGE_WORKERS", "8")), every=200)
    outcomes = collapse_orders(res)

    # win rate against THIS field (ties 0.5), averaged over nodes then judges
    per = {a: {j: {} for j in judges} for a in METHODS}
    for (j, code, x, y), w in outcomes.items():
        if w is None or x not in METHODS or y not in METHODS:
            continue
        sx = 1.0 if w == x else (0.5 if w == "tie" else 0.0)
        per[x][j].setdefault(code, []).append(sx)
        per[y][j].setdefault(code, []).append(1.0 - sx)

    summary = {}
    for m in METHODS:
        node_means, judge_means = {}, {}
        for j in judges:
            vals = {c: st.mean(v) for c, v in per[m][j].items()}
            judge_means[j] = round(st.mean(vals.values()), 4) if vals else None
            for c, v in vals.items():
                node_means.setdefault(c, []).append(v)
        nm = [st.mean(v) for v in node_means.values()]
        summary[m] = {"win_rate": round(st.mean(nm), 4) if nm else None,
                      "boot_sd": round(boot(nm), 4), "per_judge": judge_means,
                      "n_nodes": len(nm)}

    # calibration: the true label must beat every arm, and tie with itself.
    gt_beats = {}
    for m in METHODS:
        got = [w for (j, c, x, y), w in outcomes.items()
               if {x, y} == {GT, m} and w is not None]
        n = len(got) or 1
        gt_beats[m] = {"gt_wins": round(sum(w == GT for w in got) / n, 4),
                       "tie": round(sum(w == "tie" for w in got) / n, 4),
                       "method_wins": round(sum(w == m for w in got) / n, 4),
                       "n": len(got)}
    ident = [w for (j, c, x, y), w in outcomes.items() if x == GT and y == GT and w]
    control = {"gt_vs_method": gt_beats,
               "gt_vs_gt_tie_rate": round(sum(w == "tie" for w in ident) / len(ident), 4)
               if ident else None}

    # head-to-head per unordered pair: depends only on the two arms, so these
    # rates ARE comparable with label_eval_metric4.json's.
    h2h = {}
    for (j, code, x, y), w in outcomes.items():
        if w is None or GT in (x, y):
            continue
        k = "|".join(sorted((x, y)))
        d = h2h.setdefault(k, {"n": 0, "tie": 0})
        d["n"] += 1
        if w == "tie":
            d["tie"] += 1
        else:
            d[w] = d.get(w, 0) + 1
    for k, d in h2h.items():
        a, b = k.split("|")
        d["rate_" + a] = round((d.get(a, 0) + 0.5 * d["tie"]) / d["n"], 4)
        d["rate_" + b] = round((d.get(b, 0) + 0.5 * d["tie"]) / d["n"], 4)

    errs = sum(1 for v in outcomes.values() if v is None)
    out = {"judges": judges, "n_nodes": len(nodes), "field": METHODS,
           "new_arms": sorted(NEW_ARMS), "control": control, "methods": summary,
           "head_to_head": h2h, "n_pair_decisions": len(outcomes), "n_errors": errs}
    OUT.write_text(json.dumps(out, indent=2))

    print("\nCalibration (must hold before reading the scores):")
    print(f"  {'arm':10s} {'gt wins':>8s} {'tie':>8s} {'METHOD WINS':>12s}  "
          "(method wins must be ~0)")
    for m in METHODS:
        g = gt_beats[m]
        print(f"  {m:10s} {g['gt_wins']:8.3f} {g['tie']:8.3f} {g['method_wins']:12.3f}")
    print(f"  true label vs itself, tie rate: {control['gt_vs_gt_tie_rate']} "
          "(should be ~1.0)")
    print(f"\nWin rate against the field {METHODS} (ties = 0.5).")
    print("   FIELD-DEPENDENT: not comparable with label_eval_metric4.json's column.")
    for m, r in sorted(summary.items(), key=lambda kv: -(kv[1]["win_rate"] or 0)):
        print(f"  {m:12s} {r['win_rate']:.3f}  (+-{r['boot_sd']:.3f})")
    print("\nHead-to-head vs the BERTopic arms (stable under field changes):")
    for k in sorted(h2h):
        a, b = k.split("|")
        if not ({a, b} & NEW_ARMS):
            continue
        d = h2h[k]
        print(f"  {a:12s} {d['rate_' + a]:.3f}  vs  {b:12s} {d['rate_' + b]:.3f}"
              f"   (n={d['n']}, ties={d['tie']})")
    print(f"\n{errs} errored pair decisions | wrote {OUT.name}")


if __name__ == "__main__":
    main()
