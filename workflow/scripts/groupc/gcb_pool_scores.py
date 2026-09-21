"""[GPU-light] Paired per-unit score pools for the Group C similarity re-runs (#69 / #93 / #72).

Same construction as workflow/scripts/score_pool_field.py -- identical evaluation units
across methods, so the bootstrap is paired -- but the method list is open, so it can carry:
  * the 2024-era encoders (#69),
  * every text baseline WITH the same citation-trained bijective transform applied (#93's control),
  * transforms trained on pre-cutoff citations only (#72),
alongside the raw gene, the frozen genkron and the raw text baselines.

Writes into data/groupc/bench/pools/ -- a separate directory from the manuscript's pools, so Table 1
keeps its byte-identical inputs and this table is additive.

Pools: collab_{field}.parquet, np_{field}.parquet, topic_{field}.parquet, topic_temporal_{field}.parquet
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "workflow/scripts")
sys.path.insert(0, "data/layer_bands")
from bench_data import load  # noqa: E402
from eval_collab2 import author_cos  # noqa: E402

FIELD = snakemake.wildcards.field                    # noqa: F821
METHODS = dict(snakemake.params.methods)             # noqa: F821  {name: [path, key]}
OUT = snakemake.params.out_dir                       # noqa: F821
TOPIC_N = int(snakemake.params.topic_n)              # noqa: F821
TEMP_SPLIT = int(snakemake.params.temporal_split)    # noqa: F821  train <= year, test > year
WINMAP = {"aps": [2000, 2004, 2008], "economics": [2008, 2012, 2016],
          "psychology": [2008, 2012, 2016]}
DY, N_AUTH, N_NEG, SEEDS = 3, 4000, 10, [0, 1]
os.makedirs(OUT, exist_ok=True)

pt, edges, emb_dir = load(FIELD)
yr = dict(zip(pt.paper_id.astype(int), pt.year))
edges = edges.copy()
edges["year"] = edges.paper_id.map(yr)
edges = edges.dropna(subset=["year"])
edges["year"] = edges.year.astype(int)

EMB = {}
for name, (path, key) in METHODS.items():
    if not os.path.exists(path):
        print(f"  skip {name}: {path} absent", flush=True)
        continue
    z = np.load(path, allow_pickle=True)
    k = key if key in z.files else ("embeddings" if "embeddings" in z.files else "vecs")
    E = np.asarray(z[k], dtype=np.float32)
    if E.ndim > 2:
        E = E.reshape(E.shape[0], -1)
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    idk = "paper_ids" if "paper_ids" in z.files else "pids"
    EMB[name] = (E, {int(p): i for i, p in enumerate(z[idk])})
    print(f"  {name}: {E.shape}", flush=True)
methods = list(EMB)
common = set.intersection(*[set(pr) for _, pr in EMB.values()])
print(f"[gcb-pool {FIELD}] {len(methods)} methods, {len(common):,} papers covered by all", flush=True)

windows = WINMAP[FIELD]

# ── collab: candidate author pairs ──────────────────────────────────────────────────────
d = pd.read_parquet(f"data/collab_scores_{FIELD}_hard.parquet").reset_index(drop=True)
perwin = {t: edges[edges.year.between(t - DY, t)].groupby("author_id")["paper_id"]
          .apply(lambda s: tuple(int(x) for x in s)).to_dict() for t in windows}
out = d[["window", "a", "b", "y"]].copy()
for name in methods:
    E, prow = EMB[name]
    col = np.full(len(d), np.nan, np.float32)
    for t in windows:
        sub = d[d.window == t]
        cand = list(zip(sub.a.astype(int), sub.b.astype(int)))
        s, cov = author_cos(cand, perwin[t], E, prow)
        col[sub.index.values] = np.where(cov, s, np.nan)
    out[name] = col
out.to_parquet(f"{OUT}/collab_{FIELD}.parquet")
print(f"  collab pool: {len(out):,} pairs", flush=True)

# ── next paper: author cohorts ─────────────────────────────────────────────────────────
rows = []
qid = 0
for t in windows:
    past = edges[edges.year.between(t - DY, t)]
    fut = edges[edges.year.between(t + 1, t + DY)]
    past_au = past.groupby("author_id")["paper_id"].apply(
        lambda s: [int(x) for x in s if int(x) in common]).to_dict()
    fut_au = fut.groupby("author_id")["paper_id"].apply(
        lambda s: [int(x) for x in s if int(x) in common]).to_dict()
    fut_pool = np.array(sorted({p for ps in fut_au.values() for p in ps}))
    cohort = [(au, past_au[au], fut_au[au]) for au in past_au
              if au in fut_au and past_au[au] and fut_au[au]]
    for seed in SEEDS:
        rng = np.random.default_rng(seed * 100 + t)
        rng.shuffle(cohort)
        for au, ppast, pfut in cohort[:N_AUTH]:
            negs = [int(x) for x in rng.choice(fut_pool, N_NEG * 3) if int(x) not in set(pfut)][:N_NEG]
            if len(negs) < 5:
                continue
            cands = [pfut[0]] + negs
            isp = [1] + [0] * len(negs)
            sc = {}
            for name in methods:
                E, prow = EMB[name]
                v = E[[prow[p] for p in ppast]].mean(0)
                v /= (np.linalg.norm(v) + 1e-9)
                sc[name] = (E[[prow[c] for c in cands]] @ v).tolist()
            for k2, (c, ip) in enumerate(zip(cands, isp)):
                rows.append({"query_idx": qid, "window": t, "cand_pid": c, "is_pos": ip,
                             **{m: sc[m][k2] for m in methods}})
            qid += 1
pd.DataFrame(rows).to_parquet(f"{OUT}/np_{FIELD}.parquet")
print(f"  np pool: {qid:,} queries", flush=True)

# ── topic: kNN, random split and temporal split ────────────────────────────────────────
if FIELD == "aps":
    jt = pd.read_csv(snakemake.params.aps_table, usecols=["paper_id", "journal_code"]).dropna()  # noqa: F821
    codes = {c: i for i, c in enumerate(sorted(jt.journal_code.unique()))}
    lab = {int(p): codes[c] for p, c in zip(jt.paper_id, jt.journal_code)}
else:
    tp = pd.read_parquet(f"data/fields/{FIELD}/paper_topics.parquet")
    lab = dict(zip(tp.paper_id.astype(int), tp.main_class.astype(int)))

dev = "cuda" if torch.cuda.is_available() else "cpu"


def knn_pool(train_pids, test_pids, tag):
    top = pd.DataFrame({"paper_id": test_pids,
                        "true": [lab[int(p)] for p in test_pids],
                        "year": [yr.get(int(p), np.nan) for p in test_pids]})
    for name in methods:
        E, prow = EMB[name]
        Etr = torch.tensor(E[[prow[p] for p in train_pids]], device=dev)
        Ete = torch.tensor(E[[prow[p] for p in test_pids]], device=dev)
        ytr = torch.tensor([lab[int(p)] for p in train_pids], device=dev)
        pred = np.empty(len(test_pids), np.int64)
        for i in range(0, len(test_pids), 2048):
            s = Ete[i:i + 2048] @ Etr.t()
            nn = s.topk(10, dim=1).indices
            pred[i:i + nn.shape[0]] = torch.mode(ytr[nn], dim=1).values.cpu().numpy()
        top[name] = pred
    top.to_parquet(f"{OUT}/{tag}_{FIELD}.parquet")
    print(f"  {tag} pool: {len(train_pids):,} train / {len(test_pids):,} test", flush=True)


cand = np.array([p for p in common if p in lab])
rng = np.random.default_rng(0)
rng.shuffle(cand)
cand = cand[:TOPIC_N]
ntr = int(0.8 * len(cand))
knn_pool(cand[:ntr], cand[ntr:], "topic")

# temporal split: train on <= TEMP_SPLIT, test on > TEMP_SPLIT (App. B uses a random 80/20 split,
# which lets near-duplicate temporal neighbours inflate the score -- issue #72)
years = np.array([yr.get(int(p), np.nan) for p in cand], dtype=float)
tr = cand[np.nan_to_num(years, nan=-1) <= TEMP_SPLIT]
te = cand[np.nan_to_num(years, nan=-1) > TEMP_SPLIT]
if len(tr) > 100 and len(te) > 100:
    knn_pool(tr, te, "topic_temporal")
else:
    print(f"  topic_temporal skipped (train {len(tr)}, test {len(te)})", flush=True)

with open(snakemake.output.done, "w") as fh:                     # noqa: F821
    json.dump({"field": FIELD, "methods": methods, "n_common": len(common),
               "temporal_split": TEMP_SPLIT, "windows": windows,
               "n_train_temporal": int(len(tr)), "n_test_temporal": int(len(te))}, fh, indent=1)
print(f"[gcb-pool {FIELD}] done", flush=True)
