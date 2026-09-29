"""[CPU] FULL-SCALE next-paper prediction head-to-head: doc2lora layer BANDS (qwen/gemma) vs text
baselines (sbert/specter2/instructor), all fields. Author position = mean of PAST [t-dy,t] papers;
positive = a FUTURE [t+1,t+dy] paper by the same author; negatives = random future papers by others.
AUC of cosine(position, candidate) ranking pos vs negs, pooled over authors x windows x seeds.
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "workflow/scripts")
from bench_data import load

FIELDS = ["aps", "economics", "psychology", "arxiv_math", "arxiv_cs"]
WINDOWS = {"aps": [2000, 2004, 2008], "economics": [2008, 2012, 2016], "psychology": [2008, 2012, 2016],
           "arxiv_math": [2010, 2014, 2018], "arxiv_cs": [2010, 2014, 2018]}
GENES = {"qwen": "qwen_norm_lora_emb.npz", "gemma": "gemma_norm_lora_emb.npz"}
TEXT = {"sbert": ("sbert_allmpnet.npz", "vecs"), "specter2": ("baseline_specter2.npz", "vecs"),
        "instructor": ("baseline_instructor.npz", "vecs")}
DY = 3
N_AUTH = 4000
N_NEG = 10
SEEDS = [0, 1]


def auc_for(matrix, prow, past_au, fut_au, fut_pool, windows_authors):
    ys, ss = [], []
    for seed, alist in windows_authors:
        rng = np.random.default_rng(seed)
        for a, ppast, pfut in alist:
            negs = [int(x) for x in rng.choice(fut_pool, N_NEG * 3) if int(x) not in pfut][:N_NEG]
            if len(negs) < 5:
                continue
            v = matrix[ppast].mean(0); v = v / (np.linalg.norm(v) + 1e-9)
            cand = [pfut_list[0]] if False else None
            posid = pfut
            cand = [list(pfut)[0]] + negs
            M = matrix[[prow[c] for c in cand]]
            M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
            sc = M @ v
            ys += [1] + [0] * len(negs); ss += list(sc)
    return roc_auc_score(ys, ss) if len(set(ys)) > 1 else float("nan")


def build_authors(field, edges, prow):
    """Per field: list of (seed, [(author, past_rows, fut_paperids), ...]) and the fut candidate pool."""
    out = []
    fut_pool_all = set()
    # Optional FIXED cohort authors (env NP_COHORT_FILE: columns key, author_id). When set, the cohort
    # per (seed,window) is exactly the listed authors in listed order -- coverage-INDEPENDENT, so the
    # subset (bench) and full runs build identical cohorts. Without it the cohort is shuffle+capped over
    # whatever the current prow covers, which differs across coverages.
    _cf = os.environ.get("NP_COHORT_FILE")
    cmap = None
    if _cf:
        cdf = pd.read_parquet(_cf)
        cmap = {k: list(g.author_id.astype(int)) for k, g in cdf.groupby("key")}
    for t in WINDOWS[field]:
        past = edges[edges["year"].between(t - DY, t)]
        fut = edges[edges["year"].between(t + 1, t + DY)]
        past_au = past.groupby("author_id")["paper_id"].apply(lambda s: [int(x) for x in s]).to_dict()
        fut_au = fut.groupby("author_id")["paper_id"].apply(lambda s: [int(x) for x in s]).to_dict()
        for ps in fut_au.values():
            fut_pool_all.update(int(p) for p in ps if int(p) in prow)

        def build_one(a):
            if a not in past_au or a not in fut_au:
                return None
            ppast = [prow[p] for p in past_au[a] if p in prow]
            pfut = [p for p in fut_au[a] if p in prow]
            return (a, ppast, set(pfut)) if (ppast and pfut) else None

        if cmap is not None:
            for seed in SEEDS:
                key = seed * 100 + t
                alist = [c for c in (build_one(a) for a in cmap.get(key, [])) if c is not None]
                out.append((key, alist))
        else:
            cohort = [c for c in (build_one(a) for a in past_au) if c is not None]
            for seed in SEEDS:
                rng = np.random.default_rng(seed * 100 + t)
                rng.shuffle(cohort)
                out.append((seed * 100 + t, cohort[:N_AUTH]))
    # Optional FIXED negative pool (env NP_FUT_POOL_FILE): restrict the random-negative candidate set
    # to a precomputed subset, identical for every method (used by the ICAE-subset run so we only embed
    # those negatives). Random negatives from a fixed random subset are statistically equivalent.
    _fpf = os.environ.get("NP_FUT_POOL_FILE")
    if _fpf:
        fixed = set(int(p) for p in pd.read_parquet(_fpf)["paper_id"].values)
        fut_pool_all = {p for p in fut_pool_all if p in fixed}
    return out, np.array(sorted(fut_pool_all))


def score_auc(matrix, prow, cohorts, fut_pool):
    ys, ss = [], []
    for seed, alist in cohorts:
        rng = np.random.default_rng(seed)
        for a, ppast, pfut in alist:
            negs = [int(x) for x in rng.choice(fut_pool, N_NEG * 3) if int(x) not in pfut][:N_NEG]
            if len(negs) < 5:
                continue
            v = matrix[ppast].mean(0); v = v / (np.linalg.norm(v) + 1e-9)
            cand = [next(iter(pfut))] + negs
            M = matrix[[prow[c] for c in cand]]
            M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
            sc = M @ v
            ys += [1] + [0] * len(negs); ss += list(sc)
    return roc_auc_score(ys, ss) if len(set(ys)) > 1 else float("nan")


def main():
    for field in FIELDS:
        pt, edges, emb_dir = load(field)
        yr = dict(zip(pt["paper_id"].astype(int), pt["year"]))
        edges = edges.copy(); edges["year"] = edges["paper_id"].map(yr)
        edges = edges.dropna(subset=["year"]); edges["year"] = edges["year"].astype(int)
        print(f"\n===== {field}  next-paper AUC =====")
        print(f"  {'method':16}{'AUC':>8}")
        # genes (need prow per encoder; cohorts depend on prow, so rebuild per encoder)
        for enc, fn in GENES.items():
            fp = os.path.join(emb_dir, fn)
            if not os.path.exists(fp):
                continue
            z = np.load(fp, allow_pickle=True); E = z["embeddings"].astype(np.float32)
            Ln = E.shape[1] // 512; E = E.reshape(E.shape[0], Ln, 512)
            prow = {int(p): i for i, p in enumerate(z["paper_ids"])}
            cohorts, fut_pool = build_authors(field, edges, prow)
            q = Ln // 3
            for b, (lo, hi) in {"early": (0, q), "mid": (q, 2 * q), "late": (2 * q, Ln), "ALL": (0, Ln)}.items():
                Eb = np.ascontiguousarray(E[:, lo:hi, :].reshape(E.shape[0], -1))
                print(f"  {enc+'_'+b:16}{score_auc(Eb, prow, cohorts, fut_pool):>8.3f}", flush=True)
                del Eb
            del E, z
        # text baselines
        for enc, (fn, key) in TEXT.items():
            fp = os.path.join(emb_dir, fn)
            if not os.path.exists(fp):
                continue
            z = np.load(fp, allow_pickle=True); V = z[key].astype(np.float32)
            prow = {int(p): i for i, p in enumerate(z["paper_ids"])}
            cohorts, fut_pool = build_authors(field, edges, prow)
            print(f"  {enc:16}{score_auc(V, prow, cohorts, fut_pool):>8.3f}", flush=True)
            del V, z


if __name__ == "__main__":
    main()
