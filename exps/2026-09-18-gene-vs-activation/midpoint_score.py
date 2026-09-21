"""Score the midpoint decodes (#152).

Metric definitions follow `workflow/scripts/fusion_vs_copy.py` so the numbers are
comparable with the simplex work.

For a two-corner cell the copy floor collapses to a single quantity: `cos(A, B)`. That
is the min-similarity a decode would already achieve by copying ONE corner outright, so

    fusion_gain = min_sim - cos(A, B)

is positive only when the decode is closer to *both* corners than the corners are to
each other. Reporting bare `min_sim` instead would reward a midpoint that snapped to a
corner, which is exactly the failure mode being tested for.

Reported separately for L1 (near) and L5 (far), because the floor differs: near corners
are already similar, so fusion_gain is hard to earn there, and a real blend has room to
show only at L5.

Endpoints (alpha 0 and 1) are scored too. They calibrate everything else: an arm whose
*endpoints* do not resemble their own source cannot be said to have blended at the
midpoint, whatever the midpoint number says.

  python midpoint_score.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "workflow/scripts"))
from fusion_vs_copy import density, toks  # noqa: E402

SBERT_ID = "sentence-transformers/all-mpnet-base-v2"
BOOT = 2000


def boot_sd(v, rng, n=BOOT):
    v = np.asarray(v, dtype=float)
    if not len(v):
        return float("nan")
    return float(v[rng.integers(0, len(v), (n, len(v)))].mean(1).std())


def score_arm(sb, rec, rng):
    pairs = rec["pairs"]
    leads = [p["A"] for p in pairs] + [p["B"] for p in pairs]
    E = sb.encode(leads, normalize_embeddings=True, show_progress_bar=False)
    n = len(pairs)
    EA, EB = E[:n], E[n:]
    cosAB = (EA * EB).sum(1)                       # the copy floor for a 2-corner cell

    rows = []
    for a in rec["alphas"]:
        key = str(a)
        dec = [p["decodes"][key] for p in pairs]
        D = sb.encode(dec, normalize_embeddings=True, show_progress_bar=False)
        sA, sB = (D * EA).sum(1), (D * EB).sum(1)
        mins, maxs = np.minimum(sA, sB), np.maximum(sA, sB)
        gain = mins - cosAB
        dens = np.array([max(density(toks(p["A"]), toks(d)),
                             density(toks(p["B"]), toks(d)))
                         for p, d in zip(pairs, dec)])
        # clip to >0 before the ratio, as fusion_vs_copy.py does: an off-topic decode
        # gives NEGATIVE cosines, and a raw min/max on those is meaningless (it produced
        # values like -4345 on the activation arm before this clip).
        cmin, cmax = np.clip(mins, 1e-6, None), np.clip(maxs, 1e-6, None)
        bal = cmin / cmax
        for i, p in enumerate(pairs):
            rows.append(dict(set=p["set"], stratum=p["stratum"], alpha=a,
                             min_sim=float(mins[i]), copy_floor=float(cosAB[i]),
                             fusion_gain=float(gain[i]), max_density=float(dens[i]),
                             balance=float(bal[i])))
    return rows


def summarize(rows, rng):
    out = {}
    for stratum in ("L1", "L5", "all"):
        sel0 = [r for r in rows if stratum == "all" or r["stratum"] == stratum]
        for a in sorted({r["alpha"] for r in sel0}):
            sel = [r for r in sel0 if r["alpha"] == a]
            key = f"{stratum}/alpha={a}"
            out[key] = {"n": len(sel)}
            for m in ("min_sim", "copy_floor", "fusion_gain", "max_density", "balance"):
                v = [r[m] for r in sel]
                out[key][m] = float(np.mean(v))
                out[key][m + "_sd"] = boot_sd(v, rng)
            out[key]["frac_gain_positive"] = float(
                np.mean([r["fusion_gain"] > 0 for r in sel]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["act", "gene"])
    ap.add_argument("--decode-dir", default=str(HERE / "results"),
                    help="where midpoints_<arm>.json live (the T2L arm keeps its own)")
    ap.add_argument("--out", default=str(HERE / "results/midpoint_score.json"))
    a = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    sb = SentenceTransformer(SBERT_ID)
    rng = np.random.default_rng(0)
    res, raw = {}, {}
    for arm in a.arms:
        p = Path(a.decode_dir) / f"midpoints_{arm}.json"
        if not p.exists():
            print(f"[skip] {p} missing")
            continue
        rec = json.loads(p.read_text())
        rows = score_arm(sb, rec, rng)
        raw[arm] = rows
        res[arm] = summarize(rows, rng)
        print(f"\n=== {arm} ({len(rec['pairs'])} pairs) ===")
        for k in sorted(res[arm]):
            s = res[arm][k]
            print(f"  {k:<16} n={s['n']:>3}  min_sim {s['min_sim']:.3f}  "
                  f"floor {s['copy_floor']:.3f}  gain {s['fusion_gain']:+.3f}"
                  f"±{s['fusion_gain_sd']:.3f}  gain>0 {s['frac_gain_positive']:.2f}  "
                  f"density {s['max_density']:.2f}  balance {s['balance']:.2f}")

    # paired difference on the midpoint, same sets -- only when exactly the two
    # gene/act arms are present (the T2L arms are scored on their own)
    if set(a.arms) == {"act", "gene"} and len(res) == 2:
        print("\n[paired midpoint, gene - act]")
        for stratum in ("L1", "L5", "all"):
            def vec(arm):
                return {r["set"]: r["fusion_gain"] for r in raw[arm]
                        if r["alpha"] == 0.5 and (stratum == "all" or r["stratum"] == stratum)}
            g, ac = vec("gene"), vec("act")
            common = sorted(set(g) & set(ac))
            d0 = np.array([g[c] - ac[c] for c in common])
            bs = d0[rng.integers(0, len(d0), (5000, len(d0)))].mean(1)
            res.setdefault("paired", {})[stratum] = {
                "diff": float(d0.mean()), "sd": float(bs.std()),
                "p_le_0": float((bs <= 0).mean()), "n": len(common),
                "win": int((d0 > 0).sum()), "loss": int((d0 < 0).sum())}
            p = res["paired"][stratum]
            print(f"  {stratum:<4} fusion_gain {p['diff']:+.3f} ± {p['sd']:.3f}  "
                  f"P(≤0)={p['p_le_0']:.3f}  win/loss {p['win']}/{p['loss']}  n={p['n']}")

    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
