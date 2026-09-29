"""Corner-PAIR selection for the 2-corner interpolation study (L1 near ... L5 far).

Reuses sample_far_triples.py's PROVEN stratum draw (same subtopic/area/chapter/
2-chapter/3-chapter rules) so L1-L5 mean exactly what they mean elsewhere in this
project. For each valid triple draw, we take the MOST-SEPARATED of its 3 edges as
the representative pair (max SBERT distance) -- then, per the mild-sampling
recipe, over-sample 2X candidate pairs per stratum and keep the X most separated.

Writes corners_pair<L>_<NN>.json (2-key corner spec, keys=["A","B"]) + pair_manifest.json.

  python sample_far_pairs.py --per-stratum 50 --factor 2 --seed 0
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sample_far_triples import (load_universe, make_draw, corner_spec, STRATA, unit)


def best_pair(tri, E):
    """the most-separated of the 3 edges in a triple, as (pair_idx_pair, distance)."""
    d_ab = float(1 - E[tri[0]] @ E[tri[1]])
    d_ac = float(1 - E[tri[0]] @ E[tri[2]])
    d_bc = float(1 - E[tri[1]] @ E[tri[2]])
    edges = [((tri[0], tri[1]), d_ab), ((tri[0], tri[2]), d_ac), ((tri[1], tri[2]), d_bc)]
    return max(edges, key=lambda z: z[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-stratum", type=int, default=50)
    ap.add_argument("--factor", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    X = args.per_stratum
    rng = np.random.default_rng(args.seed)

    df, E = load_universe()
    draw = make_draw(df, rng)
    keys = ["A", "B"]
    manifest = []
    for L in STRATA:
        pool, seen = [], set()
        need = int(round(args.factor * X))
        tries = 0
        while len(pool) < need and tries < need * 200:
            tries += 1
            tri = draw(L)
            if tri is None:
                continue
            pair, d = best_pair(tri, E)
            key = tuple(sorted(pair))
            if key in seen:
                continue
            seen.add(key)
            pool.append((pair, d))
        pool.sort(key=lambda z: -z[1])   # keep the MOST separated pairs
        kept = pool[:X]
        print(f"{L}: sampled {len(pool)}  random-mean {np.mean([d for _,d in pool]):.3f}"
              f" -> kept-top{X} mean {np.mean([d for _,d in kept]):.3f}", flush=True)
        for nn, (pair, d) in enumerate(kept):
            s = f"pair{L}_{nn:02d}"
            rows = [df.iloc[t] for t in pair]
            corners = {k: corner_spec(r) for k, r in zip(keys, rows)}
            json.dump({"set": s, "keys": keys, "corners": corners},
                      open(os.path.join(HERE, f"corners_{s}.json"), "w"))
            manifest.append({"set": s, "stratum": L,
                             "paper_ids": [int(r["aps_paper_id"]) for r in rows],
                             "pacs": [str(r["PACS1"]) for r in rows],
                             "names": [corners[k]["name"] for k in keys],
                             "sep": round(d, 3)})
    json.dump(manifest, open(os.path.join(HERE, "pair_manifest.json"), "w"), indent=1)
    print(f"\nwrote {len(manifest)} corner pairs", flush=True)


if __name__ == "__main__":
    main()
