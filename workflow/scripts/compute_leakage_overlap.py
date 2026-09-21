"""Leakage audit for the general bijective transform g_theta (issue #22).

Reports the overlap between the transform-training citation pool and every
evaluation split. No retraining: this is a disclosure measurement, computed
purely from saved ID lists.

Training pool (this directory):
  triplets.parquet  -- 42,332 rows (a, pos, neg_easy, neg_hard); OpenAlex paper_id ints
  pool_text.parquet -- 167,111 papers (pid) that the triplets reference

Eval splits:
  Economics / Psychology  -- data/fields/{f}/paper_text.parquet, col openalex_paper_id
                             (same global OpenAlex int namespace as the pool; title-verified)
  APS                     -- data/aps/paper_text.parquet, col doi (OOD; mapped via the
                             OpenAlex master paper_table doi column)
  S2AND                   -- AllenAI release, a separate paper-id universe with no OpenAlex
                             link in the saved artifacts. The MAIN S2AND table uses the
                             in-task same-author transform, not g_theta. We report an
                             approximate normalized-title overlap as an upper bound.

Dual-mode: runs under Snakemake (rule ga_leakage_overlap) or standalone:
  ~/miniforge3/envs/doc2lora/bin/python workflow/scripts/compute_leakage_overlap.py
"""
import json
import os
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  "..", "..", "workflow", "scripts"))
from bench_data import openalex_master_paper_table  # noqa: E402
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

S2_SETS = ["zbmath", "qian", "arnetminer", "pubmed", "kisti"]


def _rel(*a):
    return os.path.join(ROOT, *a)


if "snakemake" in sys.modules:
    TRIPLETS = snakemake.input["triplets"]
    POOL_TEXT = snakemake.input["pool"]
    ECO_PT = snakemake.input["eco"]
    PSY_PT = snakemake.input["psy"]
    APS_PT = snakemake.input["aps"]
    S2_PT = dict(zip(S2_SETS, snakemake.input["s2and"]))
    OA_MASTER = snakemake.params["oa_master"]
    OUT = snakemake.output["json"]
else:
    TRIPLETS = _rel("data/general_adapter/triplets.parquet")
    POOL_TEXT = _rel("data/general_adapter/pool_text.parquet")
    ECO_PT = _rel("data/fields/economics/paper_text.parquet")
    PSY_PT = _rel("data/fields/psychology/paper_text.parquet")
    APS_PT = _rel("data/aps/paper_text.parquet")
    S2_PT = {ds: _rel("data/s2and/proc", ds, "paper_text.parquet")
             for ds in S2_SETS}
    OA_MASTER = openalex_master_paper_table()
    OUT = os.path.join(HERE, "leakage_overlap.json")


def pct(n, d):
    return 0.0 if d == 0 else 100.0 * n / d


# ── training pool ─────────────────────────────────────────────────────────
tri = pd.read_parquet(TRIPLETS)
pool = pd.read_parquet(POOL_TEXT, columns=["pid"])

POOL = set(pool.pid.tolist())                       # every paper the training touches
ANCHORS = set(tri.a.tolist())
POS = set(tri.pos.tolist())
SUPERVISED = ANCHORS | POS                          # papers in real citation edges
EDGES = set(map(tuple, tri[["a", "pos"]].drop_duplicates().itertuples(index=False)))

print(f"pool papers (any role): {len(POOL):,}")
print(f"supervised papers (anchor|positive): {len(SUPERVISED):,}")
print(f"unique citation edges (a->pos): {len(EDGES):,}")

report = {
    "pool_papers_any_role": len(POOL),
    "supervised_papers": len(SUPERVISED),
    "unique_edges": len(EDGES),
    "splits": {},
}


def field_overlap(name, parquet):
    df = pd.read_parquet(parquet, columns=["openalex_paper_id"])
    E = set(df.openalex_paper_id.tolist())
    any_role = E & POOL
    sup = E & SUPERVISED
    edges_internal = [e for e in EDGES if e[0] in E and e[1] in E]
    edges_touch = [e for e in EDGES if e[0] in E or e[1] in E]
    r = {
        "eval_papers": len(E),
        "papers_in_pool_any_role": len(any_role),
        "papers_in_pool_any_role_pct_of_eval": round(pct(len(any_role), len(E)), 4),
        "papers_in_pool_supervised": len(sup),
        "papers_in_pool_supervised_pct_of_eval": round(pct(len(sup), len(E)), 4),
        "training_edges_both_endpoints_in_eval": len(edges_internal),
        "training_edges_both_endpoints_in_eval_pct_of_edges": round(pct(len(edges_internal), len(EDGES)), 4),
        "training_edges_any_endpoint_in_eval": len(edges_touch),
        "training_edges_any_endpoint_in_eval_pct_of_edges": round(pct(len(edges_touch), len(EDGES)), 4),
    }
    report["splits"][name] = r
    print(f"\n[{name}] eval papers {len(E):,}")
    print(f"  in pool (any role)  : {len(any_role):,} ({r['papers_in_pool_any_role_pct_of_eval']}% of eval)")
    print(f"  in pool (supervised): {len(sup):,} ({r['papers_in_pool_supervised_pct_of_eval']}% of eval)")
    print(f"  training edges fully inside this field: {len(edges_internal):,} "
          f"({r['training_edges_both_endpoints_in_eval_pct_of_edges']}% of {len(EDGES):,} edges)")
    print(f"  training edges touching this field    : {len(edges_touch):,} "
          f"({r['training_edges_any_endpoint_in_eval_pct_of_edges']}% of edges)")


field_overlap("economics", ECO_PT)
field_overlap("psychology", PSY_PT)


# ── APS via DOI ───────────────────────────────────────────────────────────
def norm_doi(s):
    if not isinstance(s, str):
        return None
    s = s.strip().lower()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s)
    return s or None


print("\n[APS] mapping pool pids -> DOI via OpenAlex master paper_table (chunked)...")
pool_doi = {}            # pid -> normalized doi, for pool pids only
for chunk in pd.read_csv(OA_MASTER, usecols=["paper_id", "doi"],
                         chunksize=2_000_000, low_memory=False):
    hit = chunk[chunk.paper_id.isin(POOL)]
    for pid, d in zip(hit.paper_id.tolist(), hit.doi.tolist()):
        nd = norm_doi(d)
        if nd:
            pool_doi[pid] = nd

pool_dois = set(pool_doi.values())
sup_dois = {pool_doi[i] for i in SUPERVISED if i in pool_doi}
aps = pd.read_parquet(APS_PT, columns=["doi"])
APS = {norm_doi(d) for d in aps.doi.tolist()}
APS.discard(None)
aps_any = APS & pool_dois
aps_sup = APS & sup_dois
report["splits"]["aps"] = {
    "eval_papers": len(APS),
    "pool_papers_with_doi": len(pool_dois),
    "papers_in_pool_any_role": len(aps_any),
    "papers_in_pool_any_role_pct_of_eval": round(pct(len(aps_any), len(APS)), 4),
    "papers_in_pool_supervised": len(aps_sup),
    "papers_in_pool_supervised_pct_of_eval": round(pct(len(aps_sup), len(APS)), 4),
    "note": "APS is matched by DOI; pool pids carry a DOI only when OpenAlex records one.",
}
print(f"  APS eval DOIs {len(APS):,}; pool DOIs {len(pool_dois):,}")
print(f"  APS papers in pool (any role)  : {len(aps_any):,} "
      f"({report['splits']['aps']['papers_in_pool_any_role_pct_of_eval']}% of APS)")
print(f"  APS papers in pool (supervised): {len(aps_sup):,} "
      f"({report['splits']['aps']['papers_in_pool_supervised_pct_of_eval']}% of APS)")


# ── S2AND via normalized title (approximate upper bound) ──────────────────
def norm_title(s):
    if not isinstance(s, str):
        return None
    s = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    return s or None


print("\n[S2AND] normalized-title overlap (separate id universe; approximate)...")
pool_titles = set()
for t in pd.read_parquet(POOL_TEXT, columns=["text"]).text.tolist():
    # pool text is "Title: ...\nAbstract: ..."
    line = t.split("\n", 1)[0]
    line = re.sub(r"^title:\s*", "", line, flags=re.I)
    nt = norm_title(line)
    if nt and len(nt) > 15:        # drop trivially short / generic titles
        pool_titles.add(nt)

s2_report = {}
for ds in S2_SETS:
    fp = S2_PT[ds]
    if not os.path.exists(fp):
        s2_report[ds] = {"note": "not prepped on disk"}
        continue
    df = pd.read_parquet(fp, columns=["text"])
    titles = set()
    for t in df.text.tolist():
        nt = norm_title(str(t).split(". ", 1)[0])
        if nt and len(nt) > 15:
            titles.add(nt)
    inter = titles & pool_titles
    s2_report[ds] = {
        "eval_titles_distinct": len(titles),
        "title_overlap_with_pool": len(inter),
        "title_overlap_pct": round(pct(len(inter), len(titles)), 4),
    }
    print(f"  {ds:11s} titles {len(titles):6,}  overlap {len(inter):4,} "
          f"({s2_report[ds]['title_overlap_pct']}%)")
report["splits"]["s2and_title_overlap"] = s2_report
report["s2and_note"] = (
    "S2AND is a separate AllenAI release with no OpenAlex id link in the saved "
    "artifacts; the main S2AND table (figs/s2and_full.tex, 'doc2lora' column) uses "
    "the in-task same-author transform (60/40 train/test block split, seed 42), NOT "
    "g_theta. The general transform appears only in optional *_general.csv rows. The "
    "numbers above are an approximate normalized-title upper bound on shared papers."
)

with open(OUT, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nwrote {OUT}")
