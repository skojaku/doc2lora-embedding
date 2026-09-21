"""Corner selection (CANONICAL). Over-sample random valid triples per PACS-distance
stratum and KEEP the far half -- a mild bias toward separated corners that PRESERVES
the natural L1->L5 distance gradient (vs a max-min search, which saturates ~1.1 and
flattens the gradient).

Each kept triple is a genuine random draw under the stratum's PACS rules, ranked by
min-pairwise SBERT (all-mpnet) distance; we keep the top X of FACTOR*X sampled.

Draw rules (mirror the original stratified sampler):
  L1 same subtopic | L2 same area | L3 same chapter | L4 2 chapters (2+1) | L5 3 chapters

Writes corners_strat<NN>_mild.json (standard corner spec) + mild_manifest.json.
Set ids follow the original layout: L1->00-19, L2->20-39, ... L5->80-99.

  python sample_far_triples.py --per-stratum 20 --factor 2 --seed 0
"""
import argparse, json, os, re, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HERE_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE_ROOT, "exps", "2026-06-20-simplex-metrics"))
from sample_sci_triples import clean_title, local_words         # keyphrase / title helpers

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SBERT = os.path.join(ROOT, "data/aps/embeddings/sbert_allmpnet.npz")
PARQUET = os.path.join(ROOT, "data/aps/paper_text.parquet")
sys.path.insert(0, os.path.join(ROOT, "workflow", "scripts"))
from bench_data import aps_paper_table                        # noqa: E402
PAPER_TABLE = aps_paper_table()
PACS_RE = re.compile(r"^\d\d\.")
MIN_ABSTRACT = 600
STRATA = ["L1", "L2", "L3", "L4", "L5"]


def pacs_levels(p):
    if not isinstance(p, str) or not PACS_RE.match(p):
        return None
    return p[0], p[:2], p.split(".")[0] + "." + p.split(".")[1]


def stratum_of(codes):
    ch = {c[0] for c in codes}; ar = {c[1] for c in codes}; st = {c[2] for c in codes}
    if len(st) == 1: return "L1"
    if len(ar) == 1: return "L2"
    if len(ch) == 1: return "L3"
    if len(ch) == 2: return "L4"
    return "L5"


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def load_universe():
    z = np.load(SBERT, allow_pickle=True)
    id2row = {int(p): i for i, p in enumerate(z["paper_ids"].astype(np.int64))}
    sb_vec = z["vecs"].astype(np.float32)
    df = pd.read_parquet(PARQUET, columns=["aps_paper_id", "title", "abstract"])
    df = df[df["abstract"].fillna("").str.len() >= MIN_ABSTRACT]
    df = df[df["aps_paper_id"].isin(id2row)]
    pt = pd.read_csv(PAPER_TABLE, usecols=["paper_id", "PACS1"], dtype={"PACS1": str})
    pt = pt[pt["PACS1"].notna() & pt["PACS1"].str.match(PACS_RE, na=False)]
    df = df.merge(pt, left_on="aps_paper_id", right_on="paper_id", how="inner")
    lv = df["PACS1"].map(pacs_levels)
    df = df[lv.notna()].reset_index(drop=True)
    df["chapter"] = [pacs_levels(p)[0] for p in df["PACS1"]]
    df["area"] = [pacs_levels(p)[1] for p in df["PACS1"]]
    df["subtopic"] = [pacs_levels(p)[2] for p in df["PACS1"]]
    df["row"] = [id2row[int(p)] for p in df["aps_paper_id"]]
    print(f"universe: {len(df)} papers", flush=True)
    return df, unit(sb_vec[df["row"].to_numpy()])


def corner_spec(row):
    title = clean_title(str(row["title"]))
    abstract = str(row["abstract"])
    lead = (f"{title}. {abstract}" if title else abstract).strip()
    name = (title[:60] or " ".join(local_words(abstract)[:4]) or "physics topic")
    return {"name": name, "lead": lead, "words": local_words(f"{title}. {abstract}")}


def min_pair(tri, E):
    a, b, c = E[tri[0]], E[tri[1]], E[tri[2]]
    return float(min(1 - a @ b, 1 - a @ c, 1 - b @ c))


def make_draw(df, rng):
    by_sub = {k: np.array(v) for k, v in df.groupby("subtopic").indices.items()}
    by_area = {k: np.array(v) for k, v in df.groupby("area").indices.items()}
    by_chap = {k: np.array(v) for k, v in df.groupby("chapter").indices.items()}
    codes = df["PACS1"].to_numpy()
    subs = [s for s, v in by_sub.items() if len(v) >= 3]
    areas = [a for a, v in by_area.items() if len(v) >= 6]
    chaps = [c for c, v in by_chap.items() if len(v) >= 6]
    chaps_all = list(by_chap)

    def draw(stratum):
        for _ in range(4000):
            if stratum == "L1":
                tri = rng.choice(by_sub[subs[rng.integers(len(subs))]], 3, replace=False)
            elif stratum == "L2":
                tri = rng.choice(by_area[areas[rng.integers(len(areas))]], 3, replace=False)
            elif stratum == "L3":
                tri = rng.choice(by_chap[chaps[rng.integers(len(chaps))]], 3, replace=False)
            elif stratum == "L4":
                c2 = rng.choice(chaps_all, 2, replace=False)
                tri = np.concatenate([rng.choice(by_chap[c2[0]], 2, replace=False),
                                      rng.choice(by_chap[c2[1]], 1, replace=False)])
            else:  # L5
                c3 = rng.choice(chaps_all, 3, replace=False)
                tri = np.array([rng.choice(by_chap[c], 1)[0] for c in c3])
            if stratum_of([pacs_levels(codes[t]) for t in tri]) == stratum:
                return [int(x) for x in tri]
        return None

    return draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-stratum", type=int, default=20)
    ap.add_argument("--factor", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    X = args.per_stratum
    rng = np.random.default_rng(args.seed)

    df, E = load_universe()
    draw = make_draw(df, rng)
    keys = ["A", "B", "C"]
    manifest = []
    nn = 0
    for L in STRATA:
        pool, seen = [], set()
        need = int(round(args.factor * X))
        tries = 0
        while len(pool) < need and tries < need * 200:
            tries += 1
            tri = draw(L)
            if tri is None:
                continue
            key = tuple(sorted(tri))
            if key in seen:
                continue
            seen.add(key)
            pool.append((tri, min_pair(tri, E)))
        pool.sort(key=lambda z: -z[1])
        kept = pool[:X]
        print(f"{L}: sampled {len(pool)}  random-mean {np.mean([d for _,d in pool]):.3f}"
              f" -> kept-top{X} mean {np.mean([d for _,d in kept]):.3f}", flush=True)
        for tri, d in kept:
            s = f"strat{nn:02d}"; nn += 1
            rows = [df.iloc[t] for t in tri]
            corners = {k: corner_spec(r) for k, r in zip(keys, rows)}
            json.dump({"set": s, "keys": keys, "corners": corners},
                      open(os.path.join(HERE, f"corners_{s}_mild.json"), "w"))
            manifest.append({"set": s, "stratum": L,
                             "paper_ids": [int(r["aps_paper_id"]) for r in rows],
                             "pacs": [str(r["PACS1"]) for r in rows],
                             "names": [corners[k]["name"] for k in keys],
                             "minpair": round(d, 3)})
    json.dump(manifest, open(os.path.join(HERE, "mild_manifest.json"), "w"), indent=1)
    print(f"\nwrote {len(manifest)} mild corner sets", flush=True)


if __name__ == "__main__":
    main()
