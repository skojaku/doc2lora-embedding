"""[CPU] Build the incoherent-cluster control set for #101.

Objection: if abstraction level is set by the norm of a vector, and a centroid's norm is set by how
mutually aligned its members are, then a cluster of *unrelated* papers also produces a confident
broad field name -- so the PACS labels might be reporting dispersion rather than shared content.

The control: clusters with MATCHED CARDINALITY and (after scaling) MATCHED CENTROID NORM but no
shared topic -- members drawn across PACS main classes.  Two independent draws per real node.

Out: data/groupc/incoherent/clusters.json
  {"real":    [{node, level, n_used, members:[pid]}],           # fresh member sample per real node
   "control": [{cid, match_node, n_used, members:[pid], chapters:{cls: k}}]}
"""
import json
import os

import numpy as np
import pandas as pd

pg = pd.read_parquet(snakemake.input.paper_groups)          # noqa: F821
means = np.load(snakemake.params.means, allow_pickle=True)   # noqa: F821
texts_ok = set(pd.read_parquet(snakemake.input.paper_text).pipe(
    lambda d: d["aps_paper_id" if "aps_paper_id" in d.columns else "paper_id"]).astype(int))

N_DRAWS = int(snakemake.params.n_draws)                     # noqa: F821
SEED = int(snakemake.params.seed)                           # noqa: F821
OUT = snakemake.output.clusters                             # noqa: F821

nodes = [str(x) for x in means["nodes"]]
n_used = {k: int(v) for k, v in zip(nodes, means["n_used"])}
pg = pg[pg.paper_id.astype(int).isin(texts_ok)]
# a few rows carry a missing main class; they can still belong to a division/subdivision node, but
# they cannot seed a cross-chapter draw, so drop them from the chapter pools
pg = pg[pg.main_class_id.notna()].copy()
pg["main_class_id"] = pg.main_class_id.astype(int)
print(f"[incoh] {len(pg):,} PACS-tagged APS papers with text, {len(nodes)} real nodes", flush=True)

MAIN = sorted(int(c) for c in pg.main_class_id.unique())


def level_of(code: str) -> str:
    """'root' | 'main' (single-digit class id) | 'division' (2 digits) | 'subdivision' (dd.dd)."""
    if code == "root":
        return "root"
    if "." in code:
        return "subdivision"
    return "division" if len(code) == 2 else "main"


def members_of(code: str) -> np.ndarray:
    if code == "root":
        m = pg.paper_id.values
    elif "." in code:
        m = pg[pg.subdivision.astype(str) == code].paper_id.values
    elif len(code) == 2:
        m = pg[pg.division.astype(str) == code].paper_id.values
    else:
        m = pg[pg.main_class_id.astype(str) == code].paper_id.values
    return m.astype(np.int64)


rng = np.random.default_rng(SEED)
real, control = [], []
by_main = {c: pg[pg.main_class_id == c].paper_id.values.astype(np.int64) for c in MAIN}

for code in nodes:
    n = n_used[code]
    pool = members_of(code)
    if len(pool) == 0:
        print(f"  [{code}] no members, skipped", flush=True)
        continue
    take = pool if len(pool) <= n else rng.choice(pool, size=n, replace=False)
    real.append({"node": code, "level": level_of(code), "n_used": int(len(take)),
                 "members": [int(x) for x in take]})

    own_main = set()
    if code != "root":
        own_main = set(pg[pg.paper_id.isin(pool)].main_class_id.unique().tolist())
    # cross-chapter draw: equal share from every main class OTHER than the node's own
    pools = [c for c in MAIN if c not in own_main] or list(MAIN)
    for d in range(N_DRAWS):
        per = int(np.ceil(n / len(pools)))
        picks, chapters = [], {}
        for c in pools:
            src = by_main[c]
            k = min(per, len(src))
            sel = rng.choice(src, size=k, replace=False)
            picks.append(sel)
            chapters[str(c)] = int(k)
        allp = np.concatenate(picks)
        if len(allp) > n:
            keep = rng.choice(len(allp), size=n, replace=False)
            allp = allp[keep]
        control.append({"cid": f"{code}~ctrl{d}", "match_node": code, "n_used": int(len(allp)),
                        "members": [int(x) for x in allp], "chapters": chapters,
                        "excluded_main": sorted(int(c) for c in own_main)})
    print(f"  [{code}] real n={len(take):,}  control n={control[-1]['n_used']:,} "
          f"over {len(pools)} chapters", flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fh:
    json.dump({"real": real, "control": control, "seed": SEED, "n_draws": N_DRAWS}, fh)
print(f"[incoh] wrote {OUT}: {len(real)} real + {len(control)} control clusters", flush=True)
