"""[CPU] Compute the MINIMAL set of paper_ids a field/APS benchmark actually touches, so we
only ICAE-embed that subset (full field genes are 565k-987k papers; embedding all is unrealistic).

For one field the three tasks in eval_all.py touch:
  - topic : a FIXED 60k subsample of papers covered by ALL methods (3 genes + 3 text baselines) and
            labeled. Fixing it (vs re-sampling from the live coverage intersection) is what lets ICAE
            cover exactly those 60k; eval_all reads it back via TOPIC_IDS_FILE so every method scores
            the identical set.
  - np    : next-paper cohorts (build_authors) -> author past+future papers PLUS the full future-pool
            (score_auc draws random negatives from it), unioned over both encoder gene coverages.
  - collab: per-window author centroids (author_cos) -> all papers of the candidate-pair authors in
            the [t-DY, t] past window.

Outputs (per field):
  {field}_topic_ids.parquet   (paper_id)      -- the fixed topic subsample
  {field}_eval_ids.parquet    (paper_id,role) -- full union to ICAE-embed

Usage: python collect_field_ids.py [field ...]   (default: economics psychology aps)
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, "workflow/scripts")
sys.path.insert(0, "data/layer_bands")
from bench_data import load, aps_paper_table
from nextpaper_layerbands import build_authors, WINDOWS, N_AUTH, SEEDS, DY

OUT = "data/icae"
ENCS = ["gemma", "qwen", "mistral"]
TOPIC_N = 60000
NP_POOL = 50000        # fixed random next-paper negative pool (was the full future corpus, ~200-400k)


def npz_ids(path):
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=True)
    return set(int(p) for p in z["paper_ids"])


def labels_for(field, emb_dir):
    if field == "aps":
        jt = pd.read_csv(aps_paper_table(),
                         usecols=["paper_id", "journal_code"]).dropna()
        return set(int(p) for p in jt.paper_id)
    tp = pd.read_parquet(f"data/fields/{field}/paper_topics.parquet")
    return set(int(p) for p in tp.paper_id)


def collect(field):
    pt, edges, emb_dir = load(field)
    yr = dict(zip(pt.paper_id.astype(int), pt.year))
    edges = edges.copy()
    edges["year"] = edges.paper_id.map(yr)
    edges = edges.dropna(subset=["year"]); edges["year"] = edges.year.astype(int)

    # ---- coverage of every method that appears in eval_all (the fair-comparison set) ----
    files = [f"{e}_norm_lora_emb.npz" for e in ENCS] + \
            ["sbert_allmpnet.npz", "baseline_specter2.npz", "baseline_instructor.npz"]
    covs = {fn: npz_ids(f"{emb_dir}/{fn}") for fn in files}
    present = {fn: c for fn, c in covs.items() if c is not None}
    full_common = set.intersection(*present.values())
    labeled = labels_for(field, emb_dir)

    # ---- topic: FIXED 60k subsample of (all-method-covered & labeled), rng(0) ----
    cand = np.array(sorted(full_common & labeled))
    rng = np.random.default_rng(0); rng.shuffle(cand)
    topic_ids = set(int(p) for p in cand[:TOPIC_N])

    # ---- np: a FIXED 50k negative pool + the actual sampled-cohort papers ----
    # All future-window papers that any method can score; cap the random-negative pool at 50k (fixed).
    fut_all = set()
    for t in WINDOWS[field]:
        fut = edges[edges.year.between(t + 1, t + DY)]
        fut_all.update(int(p) for p in fut.paper_id)
    fut_all &= full_common
    fa = np.array(sorted(fut_all))
    rng2 = np.random.default_rng(1); rng2.shuffle(fa)
    fut_pool_ids = set(int(p) for p in fa[:NP_POOL])
    pd.DataFrame({"paper_id": sorted(fut_pool_ids)}).to_parquet(f"{OUT}/{field}_fut_pool.parquet")

    # cohort papers (past + positive future) for the actually-sampled <=N_AUTH authors per seed/window.
    # Pass identity prow (pid->pid) so build_authors returns pids; use the capped pool for consistency.
    os.environ["NP_FUT_POOL_FILE"] = f"{OUT}/{field}_fut_pool.parquet"
    os.environ.pop("NP_COHORT_FILE", None)               # build cohorts via shuffle on FULL coverage here
    np_ids = set(fut_pool_ids)
    prow_id = {int(p): int(p) for p in full_common}
    cohorts, _ = build_authors(field, edges, prow_id)
    crows = []
    for key, alist in cohorts:
        for a, ppast, pfut in alist:
            crows.append((int(key), int(a)))              # FIXED cohort authors -> coverage-independent np eval
            np_ids.update(int(p) for p in ppast)          # ppast holds pids (identity prow)
            np_ids.update(int(p) for p in pfut)
    pd.DataFrame(crows, columns=["key", "author_id"]).to_parquet(f"{OUT}/{field}_cohorts.parquet")
    np_ids &= full_common

    # ---- collab: papers of candidate-pair authors within each past window ----
    collab_ids = set()
    dc = pd.read_parquet(f"data/collab_scores_{field}_hard.parquet")
    dc = dc[dc.field == field] if "field" in dc.columns else dc
    for t, g in dc.groupby("window"):
        auth = set(g.a.astype(int)) | set(g.b.astype(int))
        past = edges[edges.year.between(t - DY, t)]
        collab_ids.update(int(p) for p in past[past.author_id.isin(auth)].paper_id)
    collab_ids &= full_common

    union = topic_ids | np_ids | collab_ids
    print(f"\n===== {field} =====", flush=True)
    print(f"  full corpus (pt)         : {len(pt):,}")
    print(f"  all-method common        : {len(full_common):,}  (labeled {len(full_common & labeled):,})")
    print(f"  topic (fixed)            : {len(topic_ids):,}")
    print(f"  np (cohorts+fut_pool)    : {len(np_ids):,}")
    print(f"  collab (author papers)   : {len(collab_ids):,}")
    print(f"  UNION to ICAE-embed      : {len(union):,}")

    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame({"paper_id": sorted(topic_ids)}).to_parquet(f"{OUT}/{field}_topic_ids.parquet")
    roles = {p: [] for p in union}
    for p in topic_ids: roles[p].append("topic")
    for p in np_ids: roles[p].append("np")
    for p in collab_ids: roles[p].append("collab")
    pd.DataFrame({"paper_id": list(roles), "role": ["+".join(roles[p]) for p in roles]}) \
        .to_parquet(f"{OUT}/{field}_eval_ids.parquet")
    print(f"  [saved] {OUT}/{field}_eval_ids.parquet, {OUT}/{field}_topic_ids.parquet")
    return len(union)


if __name__ == "__main__":
    fields = sys.argv[1:] or ["economics", "psychology", "aps"]
    tot = sum(collect(f) for f in fields)
    print(f"\n[TOTAL across {fields}] {tot:,} unique paper-embeddings (+167k OpenAlex pool for adapter)")
