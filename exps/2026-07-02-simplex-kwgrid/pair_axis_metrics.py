"""2-corner interpolation study: where does the decoded abstract actually LAND in
SBERT space, vs the ideal (requested) mixing weight? Simplifies the 3-corner
simplex to a single A-B axis so the decoder's discretization behavior (does it
snap to a corner, or land smoothly at the requested blend?) is directly visible.

For each pair (20 near/L1, 20 far/L5), decoded at 13 points along the A-B edge
(decode_absfollow.py / decode_absfollow_icae.py with PAIR_EDGE=1):
  ideal_t  = requested weight on corner B (0=pure A, 1=pure B)
  actual_t = least-squares projection of the decoded abstract's SBERT embedding
             onto the A->B line: dot(decoded-A, B-A) / |B-A|^2
             (can overshoot outside [0,1] -- that's meaningful, not clipped)

Figure: two horizontal number lines per (stratum, method) panel -- top = ideal
positions, bottom = mean actual position (bootstrap 95% CI), dashed connector
per point. Color = blend(red, blue) by ideal_t (which corner dominates the
request), used for BOTH the ideal and actual marker so a color/position
mismatch reads directly as decoder snapping/miscalibration.

  KWTAG=_pairaxis python pair_axis_metrics.py            # compute + plot
  KWTAG=_pairaxis python pair_axis_metrics.py --replot    # from cache, no GPU
"""
import os, sys, json
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
STRATA = ["L1", "L5"]
CACHE = os.path.join(sc.RESULTS, f"pair_axis_metrics{SUF}.json")

RED = np.array([0.85, 0.10, 0.10])
BLUE = np.array([0.15, 0.35, 0.85])


def blend(t):
    return tuple((1 - t) * RED + t * BLUE)


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def compute():
    from sentence_transformers import SentenceTransformer
    sb = SentenceTransformer("all-mpnet-base-v2", device="cuda")

    def emb(t):
        return unit(np.atleast_2d(sb.encode(t, normalize_embeddings=True, show_progress_bar=False)).ravel())

    man = {m["set"]: m for m in json.load(open(os.path.join(HERE, "pair_manifest.json")))}
    out = {}
    for s, m in man.items():
        spec = sc.load_corners(sc.corners_path(s))
        A, B = emb(spec["leads"]["A"]), emb(spec["leads"]["B"])
        axis = B - A
        denom = float(axis @ axis) + 1e-12
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
                if len(ab.split()) < 6:
                    continue
                d = emb(ab)
                actual_t = float((d - A) @ axis / denom)
                pts[ch][str(round(ideal_t, 4))] = actual_t
        out[s] = {"stratum": m["stratum"], "pts": pts}
        print(f"{s} ({m['stratum']}): " + ", ".join(f"{ch}={len(pts[ch])}/13" for ch in METHODS), flush=True)
    json.dump(out, open(CACHE, "w"))
    return out


def _boot_ci(vals, B=2000, seed=0):
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(B, len(vals)))
    means = np.asarray(vals)[idx].mean(1)
    return float(np.mean(vals)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def plot(out):
    import seaborn as sns
    sns.set_theme(style="white", font_scale=1.15)
    R = sc.GRID_DEN
    ts = [k / R for k in range(R + 1)]
    fig, axes = plt.subplots(len(STRATA), len(METHODS), figsize=(4.2 * len(METHODS), 3.0 * len(STRATA)),
                             squeeze=False)
    for r, L in enumerate(STRATA):
        sets = [s for s, v in out.items() if v["stratum"] == L]
        for c, ch in enumerate(METHODS):
            ax = axes[r][c]; ax.grid(False)
            for t in ts:
                key = str(round(t, 4))
                vals = [out[s]["pts"][ch][key] for s in sets if key in out[s]["pts"].get(ch, {})]
                if not vals:
                    continue
                mean, lo, hi = _boot_ci(vals)
                col = blend(t)
                ax.plot([t, mean], [1, 0], "--", color="0.65", lw=1.0, zorder=1)
                ax.plot([t], [1], "o", color=col, ms=9, mec="white", mew=0.6, zorder=3)
                ax.plot([mean], [0], "o", color=col, ms=9, mec="white", mew=0.6, zorder=3)
                ax.plot([lo, hi], [0, 0], "-", color=col, lw=2.5, alpha=0.5, zorder=2)
            ax.set_yticks([0, 1]); ax.set_yticklabels(["actual", "ideal"])
            ax.set_ylim(-0.35, 1.35)
            ax.set_xlim(-0.15, 1.15)
            ax.axvline(0, color="0.85", lw=0.8, zorder=0)
            ax.axvline(1, color="0.85", lw=0.8, zorder=0)
            if r == 0:
                ax.set_title(LAB[ch], fontsize=15)
            if r == len(STRATA) - 1:
                ax.set_xlabel("position on A(red) -> B(blue) axis")
            if c == 0:
                ax.text(-0.28, 0.5, L, transform=ax.transAxes, fontsize=15, fontweight="bold",
                        ha="center", va="center", rotation=90)
            ax.tick_params(labelsize=12)
            sns.despine(ax=ax, left=True)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figs", "pair_axis_landing.pdf"), bbox_inches="tight", dpi=300, transparent=True)
    plt.close(fig)
    print("saved figs/pair_axis_landing.pdf")


def plot_line(out, strata=None):
    """standard ideal-vs-actual line plot, method-colored -- one SEPARATE figure per stratum."""
    import seaborn as sns
    sns.set_theme(style="white", font_scale=1.3)
    R = sc.GRID_DEN
    ts = [k / R for k in range(R + 1)]
    ASPECT = 5.6 / 5.0   # width:height, matched to pair_axis_copyrate.py's panel aspect
    fig_w = 8.0
    for L in (strata or STRATA):
        fig, ax = plt.subplots(figsize=(fig_w, fig_w / ASPECT))
        ax.grid(False)
        sets = [s for s, v in out.items() if v["stratum"] == L]
        ax.plot([0, 1], [0, 1], "--", color="0.55", lw=1.2, zorder=1, label="ideal (y=x)")
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
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("ideal position (A -> B)", fontsize=24)
        ax.set_ylabel("actual position", fontsize=24)
        ax.legend(fontsize=20, loc="lower right", frameon=False)
        ax.set_title(L, fontsize=28)
        ax.tick_params(labelsize=24)
        sns.despine(ax=ax)
        fig.tight_layout()
        out_path = os.path.join(HERE, "figs", f"pair_axis_line_{L}.pdf")
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.5, dpi=300, transparent=True)
        plt.close(fig)
        print(f"saved figs/pair_axis_line_{L}.pdf")


def main(argv):
    if "--replot" not in argv:
        out = compute()
    else:
        out = json.load(open(CACHE))
    plot(out)
    plot_line(out)


if __name__ == "__main__":
    main(sys.argv[1:])
