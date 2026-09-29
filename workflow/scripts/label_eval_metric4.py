"""Metric 4 -- pairwise, reference-based label comparison (@skojaku, 2026-09-18).

REUSABLE PIECES. Import these rather than copying them, so a second experiment
using this design cannot drift from the one the paper reports:
    SYS, build_user   the prompt (changing either invalidates every cache entry)
    order_jobs        both orders of one pair, one job per judge
    ask_order         run one (pair, order, judge) call
    collapse_orders   both-orders-must-agree rule -> one decision per (judge, pair)
The response cache is keyed on (model, SYS, build_user(...)), i.e. on the two
labels and the reference only, so adding competitors to a tournament costs calls
for the NEW pairs alone; existing decisions are reused verbatim.


Replaces the nearest-category test, which was dropped from the paper because the
judge never saw the official label: it was shown a candidate plus three category
names and asked which one the string was nearest, so a wordier answer landed on
the specific node almost mechanically and the ordering it produced was the length
ordering.

Here the judge is SHOWN the official PACS label and two methods' labels for the
same node, and asked which of the two is closer in meaning to it. That is a
relative judgment against a visible reference rather than an absolute placement
in a hierarchy the judge has to guess.

Three design points, each aimed at a known judge failure:
  - "tie" is an explicit option. Without it a judge invents a preference when the
    two labels are equally good, which is the same failure that made "other" a
    dead option in the old metric.
  - Every pair is asked in BOTH orders, and a win counts only if both orders pick
    the same label. Disagreement is recorded as a tie. This removes position bias
    by construction instead of correcting for it afterwards.
  - Two calibration arms: the true label is entered as a competitor (it must beat
    every method), and is run against itself (must return "tie"). If either fails,
    the panel is not measuring what we think.

Scoring: win rate with ties at 0.5, averaged over nodes then over judges, with a
bootstrap sd over nodes. Agreement with the fuzzy overlap is reported too -- the
judge earns its place only where the two DISAGREE, i.e. semantic matches that a
token-set ratio misses ("Lattice QCD" vs "Quantum chromodynamics").

Reads:  label_eval_nodes.json, label_eval_metric1.json
Writes: label_eval_metric4.json
Run:    python label_eval_metric4.py            (needs OPENROUTER_API_KEY + ADC)
        JUDGES_SUBSET=glm-5.3-flash python label_eval_metric4.py   (one seat)
        CONTROL_ONLY=1 python label_eval_metric4.py                (calibration only)
"""
import itertools
import json
import os
import statistics as st
from pathlib import Path

import numpy as np

from label_eval_judges import JUDGES_V2, judge_json, run_jobs

HERE = Path(__file__).resolve().parent
from bench_data import out_dir   # bench_data.py sits next to this file
DATA = out_dir("labels")           # where this chain writes

# T2L enters through its TaskEncoder space only (the one Fig. 2 and the fusion
# comparison use); its GTE and LoRA-factor spaces decode the same few unrelated
# strings (#151). BERTopic (top 10) joined 2026-09-23 (PR #163); every pair was
# already in the judge cache from bertopic_judge.py, so this cost no calls.
METHODS = ["doc2lora", "icae", "keyllm", "bertopic", "vec2text", "incontext",
           "t2l_hidden"]
GT = "_gt"
N_BOOT = 1000

SYS = ("You compare two candidate names for a research field against the official "
       "name of that field. Judge by meaning, not by wording or length. Reply with "
       "ONLY a JSON object.")


def build_user(gt, a, b):
    return "\n".join([
        f'The official name of a field of physics is:\n  "{gt}"',
        "",
        "Two candidate labels were produced for the same cluster of papers from "
        "that field:",
        f'  A. "{a}"',
        f'  B. "{b}"',
        "",
        "Which candidate is closer in meaning to the official name?",
        'Answer "A", "B", or "tie" if the two are equally close.',
        'Return ONLY: {"choice": "A/B/tie", "reason": "<=15 words"}',
    ])


def verdict(slug, gt, a, b):
    """-> 'A', 'B', 'tie', or None on error."""
    v = judge_json(slug, SYS, build_user(gt, a, b))
    if "_error" in v:
        return None
    c = str(v.get("choice", "")).strip().upper()
    return c if c in ("A", "B") else ("tie" if c.startswith("TIE") else None)


def order_jobs(code, gt, x, y, lx, ly, judges):
    """Both orders of one pair, one job per judge. Feed to run_jobs(fn=ask_order)."""
    return [{"code": code, "gt": gt, "x": x, "y": y, "lx": lx, "ly": ly,
             "order": o, "judge": jn, "slug": sl}
            for o in (0, 1) for jn, sl in judges.items()]


def ask_order(job):
    """One (pair, order, judge) call. Order 0 shows x as A, order 1 shows y as A."""
    if job["order"] == 0:
        c = verdict(job["slug"], job["gt"], job["lx"], job["ly"])
        winner = {"A": job["x"], "B": job["y"]}.get(c, c)
    else:
        c = verdict(job["slug"], job["gt"], job["ly"], job["lx"])
        winner = {"A": job["y"], "B": job["x"]}.get(c, c)
    return {**job, "winner": winner}


def collapse_orders(results):
    """(judge, code, x, y) -> winner name, "tie", or None if a call errored.

    A win requires BOTH orders to name the same side; an order-dependent answer
    is a tie. This is where position bias is removed, so anything reusing this
    design must reuse this function rather than reimplement the rule.
    """
    bykey = {}
    for r in results:
        bykey.setdefault((r["judge"], r["code"], r["x"], r["y"]), {})[r["order"]] = r["winner"]
    out = {}
    for k, v in bykey.items():
        w0, w1 = v.get(0), v.get(1)
        if w0 is None or w1 is None:
            out[k] = None
        elif w0 == w1 and w0 in (k[2], k[3]):
            out[k] = w0
        else:
            out[k] = "tie"
    return out


def main():
    nodes = json.loads((DATA / "label_eval_nodes.json").read_text())
    judges = dict(JUDGES_V2)
    if os.environ.get("JUDGE_PANEL"):
        # "name=slug,name=slug" replaces the panel outright (smoke tests, reruns of a
        # retired roster). The reported numbers come from JUDGES_V2 with this unset.
        judges = dict(kv.split("=", 1) for kv in os.environ["JUDGE_PANEL"].split(","))
    elif os.environ.get("JUDGES_SUBSET"):
        keep = set(os.environ["JUDGES_SUBSET"].split(","))
        judges = {k: v for k, v in judges.items() if k in keep}
    control_only = bool(os.environ.get("CONTROL_ONLY"))

    # pairs: every method pair, plus each method against the true label, plus the
    # true label against itself.
    method_pairs = list(itertools.combinations(METHODS, 2))
    control_pairs = [(GT, m) for m in METHODS] + [(GT, GT)]
    pairs = control_pairs if control_only else method_pairs + control_pairs

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
    print(f"Metric 4: {len(nodes)} nodes x {len(pairs)} pairs x 2 orders x "
          f"{len(judges)} judges = {len(jobs)} calls (cached)", flush=True)

    res = run_jobs(jobs, ask_order, max_workers=8, every=200)
    outcomes = collapse_orders(res)

    def score_table(arms):
        """win rate (ties = 0.5) per arm, per judge, per node."""
        per = {a: {j: {} for j in judges} for a in arms}
        for (j, code, x, y), w in outcomes.items():
            if w is None or x not in arms or y not in arms:
                continue
            sx = 1.0 if w == x else (0.5 if w == "tie" else 0.0)
            per[x][j].setdefault(code, []).append(sx)
            per[y][j].setdefault(code, []).append(1.0 - sx)
        return per

    def boot(xs, seed=0):
        x = np.asarray(xs, dtype=float)
        if not len(x):
            return 0.0
        rng = np.random.default_rng(seed)
        return float(x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(axis=1).std())

    summary = {}
    if not control_only:
        per = score_table(METHODS)
        for m in METHODS:
            node_means, judge_means = {}, {}
            for j in judges:
                vals = {c: st.mean(v) for c, v in per[m][j].items()}
                judge_means[j] = round(st.mean(vals.values()), 4) if vals else None
                for c, v in vals.items():
                    node_means.setdefault(c, []).append(v)
            nm = [st.mean(v) for v in node_means.values()]
            summary[m] = {"win_rate": round(st.mean(nm), 4) if nm else None,
                          "boot_sd": round(boot(nm), 4),
                          "per_judge": judge_means,
                          "n_nodes": len(nm)}

    # calibration. The split matters: the true label LOSING to a method would mean
    # the panel is broken, while a TIE means the method's label is judged as close
    # to the official name as the official name itself -- a pass, and a result.
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

    # HEAD-TO-HEAD, per unordered pair. The win rate above is scored against the
    # rest of the FIELD, so it moves when the field changes: adding arms that sit
    # at the floor raises every incumbent's rate without anything about them
    # having improved. Numbers from tournaments with different arms are therefore
    # not comparable and must never share a table column. These per-pair rates
    # depend only on the two arms and are stable under adding or removing others.
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
    out = {"judges": judges, "n_nodes": len(nodes), "control": control,
           "field": METHODS, "methods": summary, "head_to_head": h2h,
           "n_pair_decisions": len(outcomes), "n_errors": errs}
    (DATA / "label_eval_metric4.json").write_text(json.dumps(out, indent=2))

    print("\nCalibration (must hold before reading the scores):")
    print(f"  {'arm':10s} {'gt wins':>8s} {'tie':>8s} {'METHOD WINS':>12s}  "
          "(method wins must be ~0)")
    for m in METHODS:
        g = gt_beats[m]
        print(f"  {m:10s} {g['gt_wins']:8.3f} {g['tie']:8.3f} {g['method_wins']:12.3f}")
    print(f"  true label vs itself, tie rate: {control['gt_vs_gt_tie_rate']} "
          "(should be ~1.0)")
    if summary:
        print(f"\nMetric 4 -- win rate against the field {METHODS} (ties = 0.5).")
        print("   FIELD-DEPENDENT: not comparable across runs with different arms.")
        for m, r in sorted(summary.items(), key=lambda kv: -(kv[1]["win_rate"] or 0)):
            print(f"  {m:10s} {r['win_rate']:.3f}  (+-{r['boot_sd']:.3f})")
        print("\nHead-to-head (stable under adding or removing other arms):")
        for k in sorted(h2h):
            a, b = k.split("|")
            d = h2h[k]
            print(f"  {a:10s} vs {b:10s}  {d['rate_'+a]:.3f} / {d['rate_'+b]:.3f}"
                  f"   (n={d['n']}, ties {d['tie']})")
    print(f"\n{errs} errored pair-decisions of {len(outcomes)}")
    print(f"wrote {HERE / 'label_eval_metric4.json'}")


if __name__ == "__main__":
    main()
