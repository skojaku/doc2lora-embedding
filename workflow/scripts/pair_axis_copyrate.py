"""Pooled verbatim copy-rate vs ideal mixing weight, across ALL corner pairs in a
stratum (L1 near / L5 far by default). Companion to pair_axis_metrics.py (which
tracks SBERT position) -- this tracks how EXTRACTIVE the decode is: fraction of
decoded-abstract word-tokens that appear verbatim in the two source lead abstracts.

  python pair_axis_copyrate.py                    # compute + plot (CPU, no GPU needed)
  python pair_axis_copyrate.py --replot            # from cache
  python pair_axis_copyrate.py --strata L1 L5 L2   # choose strata
"""
import os, sys, json, re, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simplex_common as sc
import absf_metrics as am
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SUF = os.environ.get("KWTAG", "_pairaxis")
METHODS = ["doc2lora", "incontext", "icae"]
LAB = {"doc2lora": "Doc2LoRA", "incontext": "In-context", "icae": "ICAE"}
RANDOM_BASELINE = 0.30   # empirical: unrelated-abstract token overlap (see session notes)


def toks(t):
    return re.findall(r"[a-z][a-z-]{2,}", (t or "").lower())


def cache_path(strata):
    return os.path.join(sc.RESULTS, f"pair_axis_copyrate{SUF}_{'-'.join(strata)}.json")


def compute(strata):
    man = {m["set"]: m for m in json.load(open(os.path.join(HERE, "pair_manifest.json")))
          if m["stratum"] in strata}
    out = {}
    for s, m in man.items():
        spec = sc.load_corners(sc.corners_path(s))
        srctok = set(toks(spec["leads"]["A"])) | set(toks(spec["leads"]["B"]))
        base_p = os.path.join(sc.RESULTS, f"absfollow_{s}{SUF}.json")
        icae_p = os.path.join(sc.RESULTS, f"absfollow_{s}{SUF}_icae.json")
        cellrec = {}
        if os.path.exists(base_p):
            for c in json.load(open(base_p))["cells"]:
                cellrec.setdefault(tuple(c["bary"]), {}).update(
                    {k: v for k, v in c.items() if k in ("doc2lora", "incontext")})
        if os.path.exists(icae_p):
            for c in json.load(open(icae_p))["cells"]:
                cellrec.setdefault(tuple(c["bary"]), {}).update(
                    {k: v for k, v in c.items() if k == "icae"})
        pts = {ch: {} for ch in METHODS}
        for bary, rec in cellrec.items():
            i, j = bary[0], bary[1]
            ideal_t = j / sc.GRID_DEN
            for ch in METHODS:
                ab = (rec.get(ch) or {}).get("abs", "")
                at = toks(ab)
                if len(at) < 6:
                    continue
                rate = sum(w in srctok for w in at) / len(at)
                pts[ch][str(round(ideal_t, 4))] = rate
        out[s] = {"stratum": m["stratum"], "pts": pts}
        print(f"{s} ({m['stratum']}): " + ", ".join(f"{ch}={len(pts[ch])}/13" for ch in METHODS), flush=True)
    json.dump(out, open(cache_path(strata), "w"))
    return out


def _boot_ci(vals, B=2000, seed=0):
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(B, len(vals)))
    means = np.asarray(vals)[idx].mean(1)
    return float(np.mean(vals)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def plot_line(out, strata):
    import seaborn as sns
    sns.set_theme(style="white", font_scale=1.3)
    R = sc.GRID_DEN
    ts = [k / R for k in range(R + 1)]
    fig, axes = plt.subplots(1, len(strata), figsize=(5.6 * len(strata), 5.0), squeeze=False)
    for c, L in enumerate(strata):
        ax = axes[0][c]; ax.grid(False)
        sets = [s for s, v in out.items() if v["stratum"] == L]
        n_sets = len(sets)
        ax.axhline(RANDOM_BASELINE, ls="--", color="0.55", lw=1.2, zorder=1)
        ax.text(0.55, RANDOM_BASELINE + 0.015, f"random baseline (~{int(RANDOM_BASELINE*100)}%)",
                fontsize=12, color="0.4", ha="left", va="bottom")
        for ch in METHODS:
            m, lo, hi = [], [], []
            for t in ts:
                key = str(round(t, 4))
                vals = [out[s]["pts"][ch][key] for s in sets if key in out[s]["pts"].get(ch, {})]
                a, b_, h_ = _boot_ci(vals)
                m.append(a); lo.append(b_); hi.append(h_)
            m, lo, hi = np.array(m), np.array(lo), np.array(hi)
            z = 6 if ch == "doc2lora" else 3
            lw = 3.2 if ch == "doc2lora" else 2.0
            ax.fill_between(ts, lo, hi, color=am.MCOL[ch], alpha=0.2, zorder=z - 1)
            ax.plot(ts, m, "-o", color=am.MCOL[ch], ms=5, lw=lw, label=LAB[ch], zorder=z)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(0, 0.85)
        ax.set_xlabel("ideal mixing weight (A -> B)")
        if c == 0:
            ax.set_ylabel("verbatim copy rate")
            ax.legend(fontsize=14, loc="lower left", frameon=False)
        ax.set_title(f"{L}  (n={n_sets} pairs)", fontsize=17)
        ax.tick_params(labelsize=16)
        sns.despine(ax=ax)
    fig.tight_layout()
    tag = "-".join(strata)
    out_path = os.path.join(HERE, "figs", f"pair_axis_copyrate_{tag}.pdf")
    fig.savefig(out_path, bbox_inches="tight", dpi=300, transparent=True)
    plt.close(fig)
    print(f"saved figs/pair_axis_copyrate_{tag}.pdf")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--strata", nargs="+", default=["L1", "L5"])
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args(argv)
    if not args.replot:
        out = compute(args.strata)
    else:
        out = json.load(open(cache_path(args.strata)))
    plot_line(out, args.strata)


if __name__ == "__main__":
    main(sys.argv[1:])
