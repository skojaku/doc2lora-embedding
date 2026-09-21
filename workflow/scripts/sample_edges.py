"""Sample citation TRIPLETS (anchor, positive, easy-neg, hard-neg) from the MASTER OpenAlex edge list.

Master: /data/datasets/openalex/preprocessed/{citation_net.npz, paper_table.csv, abstracts.parquet}
  citation_net: CSR over ~125M papers, ~2.56B edges; node id == paper_id (== abstracts.paper_id == paper_table.paper_id).
For a sampled positive citation a->p (a cites p): easy neg = random abstract-bearing paper; hard neg =
a paper that p cites but a does not. All four papers must carry an abstract (>= MIN_ABS chars) to be embedded.

Usage: python sample_edges.py
Env: N_TRIPLETS(100000) MAX_POOL(200000) MIN_ABS(200) OVERSAMPLE(3)
Outputs: triplets.parquet (a,pos,neg_easy,neg_hard ; ids=paper_id) + pool_text.parquet (pid, text)
"""
import os, time
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  "..", "..", "workflow", "scripts"))
from bench_data import openalex_prep_base  # noqa: E402
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc

M = openalex_prep_base()   # the full OpenAlex tables (see workflow/config.yaml)
OUT = "data/general_adapter"
N_TRIPLETS = int(os.environ.get("N_TRIPLETS", "100000"))
MAX_POOL = int(os.environ.get("MAX_POOL", "200000"))
MIN_ABS = int(os.environ.get("MIN_ABS", "200"))
OVERSAMPLE = int(os.environ.get("OVERSAMPLE", "3"))
rng = np.random.default_rng(0); t0 = time.time()


def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)


# --- 1. which papers have an abstract (read only the id column) ---
log("reading abstract paper_ids ...")
absids = ds.dataset(f"{M}/abstracts.parquet").to_table(columns=["paper_id"])["paper_id"].to_numpy()
n_nodes = 125451414
has_abs = np.zeros(n_nodes, bool); has_abs[absids[absids < n_nodes]] = True
abs_nodes = np.flatnonzero(has_abs)
log(f"abstract-bearing papers: {has_abs.sum():,}")

# --- 2. citation CSR (indptr + indices) ---
log("loading citation_net (indptr, indices) ...")
z = np.load(f"{M}/citation_net.npz", allow_pickle=True)
indptr = z["indptr"]; indices = z["indices"]; n_edges = len(indices)
log(f"nodes={len(indptr)-1:,} edges={n_edges:,}")

# --- 3. sample candidate triplets ---
log("sampling candidate edges ...")
need = N_TRIPLETS * OVERSAMPLE
trips = []; pool = set()
batch = need
while len(trips) < need and batch > 0:
    k = rng.integers(0, n_edges, batch * 2)
    src = np.searchsorted(indptr, k, side="right") - 1
    tgt = indices[k]
    ok = has_abs[src] & has_abs[tgt] & (src != tgt)
    src, tgt = src[ok], tgt[ok]
    for a, p in zip(src.tolist(), tgt.tolist()):
        na = indices[indptr[a]:indptr[a + 1]]
        npb = indices[indptr[p]:indptr[p + 1]]
        npb = npb[has_abs[npb]]
        if len(npb) == 0:
            continue
        naset = set(na.tolist())
        cand = npb[~np.isin(npb, na)]
        cand = cand[cand != a]
        if len(cand) == 0:
            continue
        hard = int(cand[rng.integers(0, len(cand))])
        easy = int(abs_nodes[rng.integers(0, len(abs_nodes))])
        if easy == a or easy in naset:
            continue
        new = {a, p, hard, easy} - pool
        if len(pool) + len(new) > MAX_POOL:
            continue
        pool |= {a, p, hard, easy}
        trips.append((a, p, easy, hard))
        if len(trips) >= N_TRIPLETS or len(pool) >= MAX_POOL:
            break
    log(f"  triplets={len(trips):,} pool={len(pool):,}")
    if len(pool) >= MAX_POOL or len(trips) >= N_TRIPLETS:
        break
T = pd.DataFrame(trips, columns=["a", "pos", "neg_easy", "neg_hard"])
poolids = np.array(sorted(pool))
log(f"sampled triplets={len(T):,} pool papers={len(poolids):,}")
del indices, indptr, z

# --- 4. text for pool: abstract (filtered parquet) + title (chunked csv) ---
log("fetching abstracts for pool ...")
at = ds.dataset(f"{M}/abstracts.parquet").to_table(
    columns=["paper_id", "abstract"], filter=pc.field("paper_id").isin(poolids.tolist())).to_pandas()
abstr = dict(zip(at.paper_id.values, at.abstract.values))
log(f"  got {len(abstr):,} abstracts")
log("fetching titles for pool (chunked) ...")
poolset = set(int(x) for x in poolids); title = {}
for ch in pd.read_csv(f"{M}/paper_table.csv", usecols=["paper_id", "title"], chunksize=4_000_000):
    sub = ch[ch.paper_id.isin(poolset)]
    title.update(zip(sub.paper_id.values, sub.title.fillna("").values))
log(f"  got {len(title):,} titles")

rows = []
for pid in poolids:
    ab = str(abstr.get(pid, "") or "")
    if len(ab) < MIN_ABS:
        continue
    rows.append((int(pid), ("Title: " + str(title.get(pid, "")) + "\nAbstract: " + ab).strip()))
ptext = pd.DataFrame(rows, columns=["pid", "text"])
good = set(ptext.pid.values)
T = T[T.a.isin(good) & T.pos.isin(good) & T.neg_easy.isin(good) & T.neg_hard.isin(good)].reset_index(drop=True)
T.to_parquet(f"{OUT}/triplets.parquet")
ptext[ptext.pid.isin(set(T.a) | set(T.pos) | set(T.neg_easy) | set(T.neg_hard))].to_parquet(f"{OUT}/pool_text.parquet")
log(f"[DONE] triplets={len(T):,} | pool papers={ptext.pid.isin(set(T.a)|set(T.pos)|set(T.neg_easy)|set(T.neg_hard)).sum():,}")
