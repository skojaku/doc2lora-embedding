"""[CPU+GPU] Dump PAIRED per-unit score pools for the field tasks (collab / next-paper / topic) so
uncertainty (bootstrap CIs) can be computed downstream. Same evaluation units across all methods.

Methods compared: {enc} (raw gene), {enc}_kron (per-field citation adapter), {enc}_genkron (general
OpenAlex adapter), sbert, specter2, instructor.  Bootstrap unit per task:
  collab -> candidate pair (rows); next-paper -> query_idx (cluster bootstrap); topic -> test paper.

Usage: CUDA_VISIBLE_DEVICES=0 python score_pool_field.py economics gemma
Outputs (data/uncertainty/pools):
  collab_<field>_<enc>.parquet  [window,a,b,y, <method score cols>]
  np_<field>_<enc>.parquet      [query_idx,cand_pid,is_pos, <method score cols>]
  topic_<field>_<enc>.parquet   [paper_id,true, <method pred cols>]
"""
import sys, os
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "workflow/scripts"); sys.path.insert(0, "data/layer_bands")
from bench_data import load, aps_paper_table
from eval_collab2 import author_cos

FIELD = sys.argv[1]; ENC = sys.argv[2]
from bench_data import out_dir   # bench_data.py sits next to this file
OUT = str(out_dir("uncertainty") / "pools"); os.makedirs(OUT, exist_ok=True)
WINMAP = {"aps": [2000, 2004, 2008], "economics": [2008, 2012, 2016], "psychology": [2008, 2012, 2016]}
# A corpus outside that table (the sample, or a new field) carries its own windows.
_W = os.environ.get("SCORE_WINDOWS")
DY, N_AUTH, N_NEG, SEEDS = 3, 4000, 10, [0, 1]
GENE = [(ENC, f"{ENC}_norm_lora_emb.npz"), (f"{ENC}_kron", f"{ENC}_kron_emb.npz"),
        (f"{ENC}_genkron", f"{ENC}_genkron_emb.npz")]
TEXT = [("sbert", "sbert_allmpnet.npz"), ("specter2", "baseline_specter2.npz"), ("instructor", "baseline_instructor.npz"),
        ("embeddinggemma", "baseline_embeddinggemma.npz"), ("gte", "baseline_gte.npz")]

pt, edges, emb_dir = load(FIELD)
_broot = os.environ.get("BENCH_ROOT")     # bootstrap units reproducible from the sliced subset
if _broot:
    emb_dir = f"{_broot}/{FIELD}/embeddings"
# A per-field citation Kron adapter only ever existed for economics/psychology (all encoders) and for
# APS-mistral; everywhere else its absence is expected. Every other method is mandatory -- silently
# dropping one produces a pool that looks fine and quietly rewrites the paper's table with fewer
# methods. Fail loudly instead.
OPTIONAL = {f"{ENC}_kron"} if (FIELD == "aps" and ENC != "mistral") else set()
missing = [(n, f"{emb_dir}/{f}") for n, f in GENE + TEXT
           if n not in OPTIONAL and not os.path.exists(f"{emb_dir}/{f}")]
print(f"[{FIELD}/{ENC}] emb_dir: {emb_dir}"
      f"{' (BENCH_ROOT)' if _broot else ''}", flush=True)
if missing:
    msg = (f"[{FIELD}/{ENC}] {len(missing)} required embedding(s) missing under {emb_dir}:\n"
           + "\n".join(f"    {n:16s} {p}" for n, p in missing)
           + "\n  Scoring would silently drop these methods from the pool. Check BENCH_ROOT / emb_dir, "
             "or set ALLOW_MISSING_METHODS=1 to proceed anyway (exploratory runs only).")
    if os.environ.get("ALLOW_MISSING_METHODS") != "1":
        sys.exit(msg)
    print(f"WARNING: {msg}", flush=True)

yr = dict(zip(pt.paper_id.astype(int), pt.year)); edges = edges.copy()
edges["year"] = edges.paper_id.map(yr); edges = edges.dropna(subset=["year"]); edges["year"] = edges.year.astype(int)

EMB = {}  # name -> (normalized matrix, prow)
for name, fn in GENE + TEXT:
    p = f"{emb_dir}/{fn}"
    if not os.path.exists(p):
        continue
    z = np.load(p, allow_pickle=True)
    key = "embeddings" if name.startswith(ENC) else "vecs"
    E = z[key].astype(np.float32)
    if E.ndim > 2:
        E = E.reshape(E.shape[0], -1)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    EMB[name] = (E, {int(p): i for i, p in enumerate(z["paper_ids"])})
methods = list(EMB)
print(f"[{FIELD}/{ENC}] methods: {methods}", flush=True)

windows = [int(w) for w in _W.split(",")] if _W else WINMAP[FIELD]

# ---------------- collab: per candidate pair ----------------
d = pd.read_parquet(f"data/collab_scores_{FIELD}_hard.parquet").reset_index(drop=True)
perwin = {t: edges[edges.year.between(t - DY, t)].groupby("author_id")["paper_id"]
          .apply(lambda s: tuple(int(x) for x in s)).to_dict() for t in windows}
out = d[["window", "a", "b", "y"]].copy()
for name in methods:                      # all methods via author_cos (consistent)
    E, prow = EMB[name]
    col = np.full(len(d), np.nan, np.float32)
    for t in windows:
        sub = d[d.window == t]
        cand = list(zip(sub.a.astype(int), sub.b.astype(int)))
        s, cov = author_cos(cand, perwin[t], E, prow)
        col[sub.index.values] = np.where(cov, s, np.nan)
    out[name] = col
out.to_parquet(f"{OUT}/collab_{FIELD}_{ENC}.parquet")
print(f"  collab pool: {len(out)} pairs", flush=True)

# ---------------- next-paper: paired queries over common coverage ----------------
common = set.intersection(*[set(prow) for _, prow in EMB.values()])
rows = []; qid = 0
for t in windows:
    past = edges[edges.year.between(t - DY, t)]; fut = edges[edges.year.between(t + 1, t + DY)]
    past_au = past.groupby("author_id")["paper_id"].apply(lambda s: [int(x) for x in s if int(x) in common]).to_dict()
    fut_au = fut.groupby("author_id")["paper_id"].apply(lambda s: [int(x) for x in s if int(x) in common]).to_dict()
    fut_pool = np.array(sorted({p for ps in fut_au.values() for p in ps}))
    cohort = [(a, past_au[a], fut_au[a]) for a in past_au if a in fut_au and past_au[a] and fut_au[a]]
    for seed in SEEDS:
        rng = np.random.default_rng(seed * 100 + t)
        rng.shuffle(cohort)
        for a, ppast, pfut in cohort[:N_AUTH]:
            negs = [int(x) for x in rng.choice(fut_pool, N_NEG * 3) if int(x) not in set(pfut)][:N_NEG]
            if len(negs) < 5:
                continue
            cands = [pfut[0]] + negs; isp = [1] + [0] * len(negs)
            scores = {}
            for name in methods:
                E, prow = EMB[name]
                v = E[[prow[p] for p in ppast]].mean(0); v = v / (np.linalg.norm(v) + 1e-9)
                M = E[[prow[c] for c in cands]]
                scores[name] = (M @ v).tolist()
            for k, (c, ip) in enumerate(zip(cands, isp)):
                rows.append({"query_idx": qid, "cand_pid": c, "is_pos": ip,
                             **{m: scores[m][k] for m in methods}})
            qid += 1
pd.DataFrame(rows).to_parquet(f"{OUT}/np_{FIELD}_{ENC}.parquet")
print(f"  np pool: {qid} queries", flush=True)

# ---------------- topic: per test paper, kNN pred per method ----------------
if FIELD == "aps":
    jt = pd.read_csv(aps_paper_table(), usecols=["paper_id", "journal_code"]).dropna()
    codes = {c: i for i, c in enumerate(sorted(jt.journal_code.unique()))}
    lab = {int(p): codes[c] for p, c in zip(jt.paper_id, jt.journal_code)}
else:
    tp = pd.read_parquet(f"data/fields/{FIELD}/paper_topics.parquet")
    lab = dict(zip(tp.paper_id.astype(int), tp.main_class.astype(int)))
cand = np.array([p for p in common if p in lab]); rng = np.random.default_rng(0); rng.shuffle(cand)
cand = cand[:60000]; ntr = int(0.8 * len(cand)); tr, te = cand[:ntr], cand[ntr:]
dev = "cuda" if torch.cuda.is_available() else "cpu"
top = pd.DataFrame({"paper_id": te, "true": [lab[int(p)] for p in te]})
for name in methods:
    E, prow = EMB[name]
    Etr = torch.tensor(E[[prow[p] for p in tr]], device=dev)
    Ete = torch.tensor(E[[prow[p] for p in te]], device=dev)
    ytr = torch.tensor([lab[int(p)] for p in tr], device=dev)
    pred = np.empty(len(te), np.int64)
    for i in range(0, len(te), 2048):
        sims = Ete[i:i + 2048] @ Etr.t()
        nn = sims.topk(10, dim=1).indices
        pred[i:i + nn.shape[0]] = torch.mode(ytr[nn], dim=1).values.cpu().numpy()
    top[name] = pred
top.to_parquet(f"{OUT}/topic_{FIELD}_{ENC}.parquet")
print(f"  topic pool: {len(te)} test papers\n[DONE] {FIELD}/{ENC}", flush=True)
