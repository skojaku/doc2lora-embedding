"""Score the gene-vs-activation decodes (#152).

Reads the per-arm decode files written by `decode_compare.py` so the protocol can be
revised without re-running GPU work.

M-A (primary) -- source identification
    Each decode competes to retrieve its own abstract against 99 distractors drawn from
    the **same PACS subdivision**. Random distractors previously manufactured a win on
    next-paper prediction that vanished under same-subfield ones, so the hard protocol
    is the only one reported. MRR and top-1, bootstrap s.d. over items.

M-B -- fidelity
    SBERT cosine to the source abstract, reported as a lift over the floor decode
    (nothing injected). The continuation prompt has a strong biomedical prior, so an
    absolute cosine means nothing.

Degeneracy guard
    The layer sweep exposed the failure mode this guards against: a decode reading
    `BookBookBookBook...` scored MRR 0.215 against a chance of 0.090, because SBERT
    places degenerate strings somewhere that happens to correlate with abstracts. Any
    decode whose 4-gram repetition rate is too high or whose type-token ratio is too low
    is excluded. **The same thresholds are applied to every arm**, and the number
    excluded is reported per arm -- a guard applied to one side only would be its own
    bias.

  python score_compare.py [--arms act gene]
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from bench_data import out_dir   # bench_data.py sits next to this file
DATA = out_dir("actpatch")           # where this chain writes

ROOT = HERE.parents[1]
CA = ROOT / "data/pacs"
SBERT_ID = "sentence-transformers/all-mpnet-base-v2"

N_DISTRACTORS = 99
REP_MAX = 0.30          # max share of any single 4-gram
TTR_MIN = 0.35          # min type-token ratio
BOOT = 1000


def toks(t):
    return re.findall(r"[a-z0-9]+", str(t).lower())


def degeneracy(text):
    w = toks(text)
    if len(w) < 8:
        return dict(rep_rate=1.0, ttr=0.0, degenerate=True)
    grams = [" ".join(w[i:i + 4]) for i in range(len(w) - 3)]
    rep = Counter(grams).most_common(1)[0][1] / len(grams)
    ttr = len(set(w)) / len(w)
    return dict(rep_rate=float(rep), ttr=float(ttr),
                degenerate=bool(rep > REP_MAX or ttr < TTR_MIN))


def distractor_pool(subdivisions, seed=7):
    """abstract text for every paper in the subdivisions we need distractors from."""
    pg = pd.read_parquet(CA / "results/paper_groups.parquet",
                         columns=["paper_id", "subdivision"])
    pg = pg[pg.subdivision.notna()]
    pg["subdivision"] = pg.subdivision.astype(str)
    pg = pg[pg.subdivision.isin(set(subdivisions))]
    txt = pd.read_parquet(ROOT / "data/aps/paper_text_pid.parquet",
                          columns=["paper_id", "abstract"])
    txt = txt[txt.abstract.notna() & (txt.abstract.str.len() > 200)]
    df = pg.merge(txt, on="paper_id", how="inner")
    return {s: g[["paper_id", "abstract"]].reset_index(drop=True)
            for s, g in df.groupby("subdivision")}


def boot_ci(v, rng, n=BOOT):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(v), size=(n, len(v)))
    m = v[idx].mean(1)
    return float(v.mean()), float(m.std())


def score_arm(sb, rec, pools, rng):
    items = rec["items"]
    deg = [degeneracy(it["decode"]) for it in items]
    keep = [i for i, d in enumerate(deg) if not d["degenerate"]]

    D = sb.encode([it["decode"] for it in items], normalize_embeddings=True,
                  show_progress_bar=False)
    A = sb.encode([it["abstract"] for it in items], normalize_embeddings=True,
                  show_progress_bar=False)
    f = sb.encode([rec["floor"]], normalize_embeddings=True, show_progress_bar=False)[0]

    fid = (D * A).sum(1)
    floor_fid = A @ f

    rr, hit1, n_short = [], [], 0
    for i, it in enumerate(items):
        pool = pools.get(it["subdivision"])
        if pool is None:
            continue
        cand = pool[pool.paper_id != it["paper_id"]]
        if len(cand) < N_DISTRACTORS:
            n_short += 1
            continue
        sel = cand.sample(n=N_DISTRACTORS, random_state=int(rng.integers(1 << 30)))
        E = sb.encode([it["abstract"]] + list(sel.abstract), normalize_embeddings=True,
                      show_progress_bar=False)
        s = E @ D[i]
        rank = int((s[1:] > s[0]).sum()) + 1
        rr.append(1.0 / rank)
        hit1.append(rank == 1)

    keep_set = set(keep)
    rr_k = [r for j, r in enumerate(rr) if j in keep_set]
    h_k = [h for j, h in enumerate(hit1) if j in keep_set]

    mrr, mrr_sd = boot_ci(rr, rng)
    mrr_k, mrr_k_sd = boot_ci(rr_k, rng)
    t1, t1_sd = boot_ci(hit1, rng)
    t1_k, _ = boot_ci(h_k, rng)
    return dict(
        arm=rec["arm"], n=len(items), n_scored=len(rr),
        n_degenerate=len(items) - len(keep), n_pool_short=n_short,
        mrr=mrr, mrr_sd=mrr_sd, top1=t1, top1_sd=t1_sd,
        mrr_clean=mrr_k, mrr_clean_sd=mrr_k_sd, top1_clean=t1_k,
        chance_mrr=float(np.mean([1 / r for r in range(1, N_DISTRACTORS + 2)])),
        chance_top1=1.0 / (N_DISTRACTORS + 1),
        fidelity=float(fid.mean()), floor_fidelity=float(floor_fid.mean()),
        fidelity_delta=float(fid.mean() - floor_fid.mean()),
        median_rep_rate=float(np.median([d["rep_rate"] for d in deg])),
        median_ttr=float(np.median([d["ttr"] for d in deg])),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["act", "gene"])
    ap.add_argument("--out", default=str(DATA / "score_compare.json"))
    a = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    recs = {}
    for arm in a.arms:
        p = DATA / f"decodes_{arm}.json"
        if not p.exists():
            print(f"[skip] {p} missing")
            continue
        recs[arm] = json.loads(p.read_text())
    if not recs:
        sys.exit("no decode files found")

    subs = {it["subdivision"] for r in recs.values() for it in r["items"]}
    pools = distractor_pool(subs)
    print(f"[pool] {len(pools)} subdivisions, "
          f"median size {int(np.median([len(v) for v in pools.values()]))}", flush=True)

    sb = SentenceTransformer(SBERT_ID)
    rng = np.random.default_rng(0)
    out = {}
    for arm, rec in recs.items():
        out[arm] = score_arm(sb, rec, pools, rng)
        r = out[arm]
        print(f"\n[{arm}] n={r['n']} scored={r['n_scored']} "
              f"degenerate={r['n_degenerate']} pool_short={r['n_pool_short']}")
        print(f"  M-A  MRR  {r['mrr']:.3f} +- {r['mrr_sd']:.3f}   "
              f"(chance {r['chance_mrr']:.3f})")
        print(f"       top1 {r['top1']:.3f} +- {r['top1_sd']:.3f}  "
              f"(chance {r['chance_top1']:.3f})")
        print(f"       MRR excl. degenerate {r['mrr_clean']:.3f} +- {r['mrr_clean_sd']:.3f}")
        print(f"  M-B  fidelity {r['fidelity']:.3f} "
              f"(floor {r['floor_fidelity']:.3f}, delta {r['fidelity_delta']:+.3f})")
        print(f"       median rep_rate {r['median_rep_rate']:.2f}  "
              f"median TTR {r['median_ttr']:.2f}")

    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
