"""Aggregate the fusion-smoothness curve over ALL decoded mild corner sets.

For every set with decoded abstracts, measure per-method tracking fidelity
r = corr(measured corner-weight, ideal barycentric weight) across its 91x3 points
(same as absf_metrics.plot_curve, one scalar per set/method). Group by PACS-distance
stratum -> mean +/- std, and plot r vs stratum (L1..L5) with a line per method.

  CUDA_VISIBLE_DEVICES=0 KWTAG=_kwsrc python agg_fusion_curve.py            # compute + plot
  KWTAG=_kwsrc python agg_fusion_curve.py --replot                         # from cache, no GPU
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import absf_metrics as am
import simplex_common as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _style import setup_style

HERE = am.HERE
SUF = am.SUF
METHODS = am.METHODS
STRATA = ["L1", "L2", "L3", "L4", "L5"]
CACHE = os.path.join(sc.RESULTS, f"agg_fusion_curve{SUF}.json")


RHO = {ch: 1.0 for ch in METHODS}   # per-method reweight exponent (tau* = am.TAU / RHO[ch])


def reweight(w, rho):
    """re-temperature a stored softmax weight: softmax(cos/tau) -> softmax(cos/(tau0/rho))."""
    a = np.clip(np.asarray(w, float), 1e-12, None) ** rho
    return a / a.sum()


def corner_floor(Dc):
    """per (set, method), per corner j: (floor, ceil) to min-max rescale angular distance.
    floor_j = distance from the method's OWN pure-corner-j decode to corner j (best achievable,
    paraphrase noise only). ceil_j = worst (max) distance to corner j observed anywhere in that
    set/method's cells (e.g. at the opposite corners). Rescaled d' = (d-floor)/(ceil-floor) in
    [0,1]: 0 = at-ceiling paraphrase quality, 1 = as far as this method/set ever gets. All derived
    from cached cos, no re-decode/re-embed."""
    R = sc.GRID_DEN
    corner_b = [(R, 0, 0), (0, R, 0), (0, 0, R)]
    out = {}
    for s in Dc:
        cc = Dc[s].get("cellc", {})
        out[s] = {}
        for ch in METHODS:
            floor = []
            for j, b in enumerate(corner_b):
                key = f"{ch}|{'-'.join(map(str, b))}"
                cos = cc.get(key)
                floor.append(am.angdist(np.asarray(cos))[j] if cos is not None else 0.0)
            allD = [am.angdist(np.asarray(v)) for k, v in cc.items() if k.startswith(ch + "|") and v is not None]
            ceil = np.max(np.stack(allD), axis=0) if allD else np.array(floor)
            out[s][ch] = {"floor": np.asarray(floor), "ceil": ceil}
    return out


def pooled_points(Dc, ch, rho, mode="softmax", floor=None):
    xs, ys = [], []
    for s in Dc:
        cw = Dc[s].get("cellw", {})
        cc = Dc[s].get("cellc", {})
        fl = floor[s][ch] if floor is not None else None
        for key, w in cw.items():
            if w is None or not key.startswith(ch + "|"):
                continue
            b = [int(x) for x in key.split("|")[1].split("-")]
            tot = sum(b)
            if mode == "argmax":
                wr = np.zeros(3); wr[int(np.argmax(w))] = 1.0   # hard: corner = argmax cosine
            elif mode == "idw":
                wr = am.idw(cc[key])   # -> 1 exactly as decode -> corner (raw cos, not re-temp'd)
            elif mode == "angdist":
                wr = am.angdist(cc[key])   # raw angular distance, -> 0 as decode -> corner
            elif mode == "cossim":
                wr = np.asarray(cc[key])   # raw cosine similarity, -> 1 as decode -> corner
            elif mode == "angdist_corrected":
                wr = np.clip((am.angdist(cc[key]) - fl["floor"]) / (fl["ceil"] - fl["floor"] + 1e-9), 0, 1)
            elif mode == "idw_corrected":
                dsc = np.clip((am.angdist(cc[key]) - fl["floor"]) / (fl["ceil"] - fl["floor"] + 1e-9), 0, 1)
                wr = am.idw_from_d(dsc)
            else:
                wr = reweight(w, rho)
            for j in range(3):
                xs.append(b[j] / tot); ys.append(wr[j])
    return np.array(xs), np.array(ys)


def optimize_rho(Dc, ch):
    """rho minimizing MSE(measured, ideal) for a SINGLE method's points."""
    best = None
    for rho in np.geomspace(0.15, 6.0, 120):
        xs, ys = pooled_points(Dc, ch, rho)
        mse = float(((ys - xs) ** 2).mean())
        if best is None or mse < best[1]:
            best = (rho, mse)
    return best[0]


def curve_r(cellw, ch):
    xs, ys = [], []
    for key, w in cellw.items():
        if not key.startswith(ch + "|") or w is None:
            continue
        b = [int(x) for x in key.split("|")[1].split("-")]
        tot = sum(b)
        wr = reweight(w, RHO[ch])
        for j in range(3):
            xs.append(b[j] / tot); ys.append(wr[j])
    if len(xs) < 6:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def compute():
    man = {m["set"]: m["stratum"] for m in json.load(open(os.path.join(HERE, "mild_manifest.json")))}
    present = [f"{s}_mild" for s in man
               if os.path.exists(os.path.join(sc.RESULTS, f"absfollow_{s}_mild{SUF}.json"))]
    print(f"{len(present)} sets decoded", flush=True)
    D = am.compute(present)                       # GPU: RGB responsibility per cell
    rows = {}                                     # set -> {method: r}
    for s in present:
        rows[s] = {ch: curve_r(D[s]["cellw"], ch) for ch in METHODS}
    stratum = {f"{k}_mild": v for k, v in man.items()}
    out = {"rows": rows, "stratum": stratum}
    json.dump(out, open(CACHE, "w"))
    return out


def plot(out):
    import seaborn as sns
    sns.set_theme(style="white", palette="colorblind", font_scale=1.3)
    rows, stratum = out["rows"], out["stratum"]
    fig, ax = plt.subplots(figsize=(5.8, 4.2)); ax.grid(False)
    xt = np.arange(len(STRATA))
    for ch in METHODS:
        mean, lo, hi = [], [], []
        for L in STRATA:
            vals = [rows[s][ch] for s in rows if stratum[s] == L and rows[s][ch] is not None]
            a, b, c = _boot_ci(vals)
            mean.append(a); lo.append(a - b); hi.append(c - a)
        z = 6 if ch == "doc2lora" else 3
        ax.errorbar(xt, mean, yerr=[lo, hi], marker="o", ms=6,
                    lw=3.2 if ch == "doc2lora" else 2.0, capsize=3, zorder=z,
                    color=am.MCOL[ch], label=am.LAB[ch])
    ax.set_xticks(xt); ax.set_xticklabels([f"{L}\n(n={sum(1 for s in rows if stratum[s]==L)})" for L in STRATA])
    ax.set_xlabel("PACS distance stratum (near → far)")
    ax.set_ylabel("tracking fidelity")
    ax.set_ylim(0, 1); ax.legend(fontsize=17, frameon=False, loc="lower left")
    ax.tick_params(labelsize=16)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figs", "absf_fusion_curve_all.pdf"), bbox_inches="tight", dpi=300, transparent=True)
    plt.close(fig)
    print("saved figs/absf_fusion_curve_all.pdf")
    # text summary
    for L in STRATA:
        line = f"  {L}: "
        for ch in METHODS:
            vals = [rows[s][ch] for s in rows if stratum[s] == L and rows[s][ch] is not None]
            line += f"{am.LAB[ch]} {np.mean(vals):.2f}  " if vals else f"{am.LAB[ch]} --  "
        print(line)


def _boot_ci(vals, B=2000, seed=0):
    """bootstrap 95% CI of the mean."""
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(B, len(vals)))
    means = np.asarray(vals)[idx].mean(1)
    return float(np.mean(vals)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def plot_pooled(out, cellcache, mode="softmax", fname="absf_fusion_curve_pooled"):
    """pooled ideal-vs-measured response curve over ALL sets, bootstrap 95% CI per bin.
    mode='softmax' = soft weight; mode='argmax' = P(corner is the argmax) vs ideal weight."""
    import seaborn as sns
    sns.set_theme(style="white", palette="colorblind", font_scale=1.3)
    Dc = json.load(open(cellcache))
    floor = corner_floor(Dc) if mode in ("angdist_corrected", "idw_corrected") else None
    bins = np.linspace(0, 1, 25)
    ctr = 0.5 * (bins[1:] + bins[:-1])
    fig, ax = plt.subplots(figsize=(5.6, 5.2)); ax.grid(False)
    if mode in ("angdist", "angdist_corrected"):
        ax.plot([0, 1], [1, 0], "--", color="0.55", lw=1.2, zorder=1, label="ideal (d=1-w)")
    elif mode == "cossim":
        ax.axhline(1.0, ls="--", color="0.55", lw=1.2, zorder=1, label="perfect match (cos=1)")
    else:
        ax.plot([0, 1], [0, 1], "--", color="0.55", lw=1.2, zorder=1, label="ideal (y=x)")
    ymin, ymax = 0.0, 0.0
    for ch in METHODS:
        xs, ys = pooled_points(Dc, ch, RHO[ch], mode=mode, floor=floor)
        m, lo, hi = [], [], []
        for i in range(len(bins) - 1):
            sel = ys[(xs >= bins[i]) & (xs < bins[i + 1] + 1e-9)]
            a, b_, c = _boot_ci(sel)
            m.append(a); lo.append(b_); hi.append(c)
        m, lo, hi = np.array(m), np.array(lo), np.array(hi)
        if mode in ("angdist", "angdist_corrected"):
            ymax = max(ymax, np.nanmax(hi), 1.0)
        elif mode == "cossim":
            ymin = min(ymin, np.nanmin(lo)); ymax = max(ymax, np.nanmax(hi), 1.0)
        else:
            ymax = 1.0
        z = 6 if ch == "doc2lora" else 3
        lw = 3.2 if ch == "doc2lora" else 2.0
        ax.fill_between(ctr, lo, hi, color=am.MCOL[ch], alpha=0.2, zorder=z - 1)
        ax.plot(ctr, m, "-o", color=am.MCOL[ch], ms=5, lw=lw, label=am.LAB[ch], zorder=z)
    ax.set_xlim(0, 1)
    if mode in ("angdist", "angdist_corrected"):
        ax.set_ylim(0, ymax * 1.08)
    elif mode == "cossim":
        ax.set_ylim(ymin - 0.05, ymax * 1.05)
    else:
        ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("ideal corner weight")
    dname = "angular distance (rad/π)" if am.DIST_KIND == "angular" else "cosine distance (1-cos)"
    ylab = {"argmax": "P(corner = argmax)", "idw": "measured weight (inverse-distance)",
            "angdist": f"{dname} to corner",
            "angdist_corrected": f"{dname}, min-max rescaled (0=floor, 1=ceiling)",
            "idw_corrected": "measured weight (rescaled-distance IDW)",
            "cossim": "cosine similarity to corner"}.get(mode, "measured corner weight")
    ax.set_ylabel(ylab)
    ax.legend(fontsize=17, loc=("upper right" if mode in ("angdist", "angdist_corrected") else "upper left"), frameon=False)
    ax.tick_params(labelsize=16)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figs", f"{fname}.pdf"), bbox_inches="tight", dpi=300, transparent=True)
    plt.close(fig)
    print(f"saved figs/{fname}.pdf")


def main(argv):
    global RHO
    cellcache = os.path.join(sc.RESULTS, f"absf_metrics{SUF}.json")
    man = {m["set"]: m["stratum"] for m in json.load(open(os.path.join(HERE, "mild_manifest.json")))}
    if "--replot" not in argv:
        present = [f"{s}_mild" for s in man
                   if os.path.exists(os.path.join(sc.RESULTS, f"absfollow_{s}_mild{SUF}.json"))]
        print(f"{len(present)} sets decoded", flush=True)
        am.compute(present)                      # GPU: (re)build per-cell responsibilities
    Dc = json.load(open(cellcache))
    # optimize softmax temperature PER METHOD, purely from cached weights (no re-embed / re-decode)
    for ch in METHODS:
        RHO[ch] = 1.0 if "--no-opt" in argv else optimize_rho(Dc, ch)
        print(f"  {am.LAB[ch]:<11} rho={RHO[ch]:.3f}  tau*={am.TAU / RHO[ch]:.3f}", flush=True)
    stratum = {s: man[s.replace('_mild', '')] for s in Dc}
    rows = {s: {ch: curve_r(Dc[s]["cellw"], ch) for ch in METHODS} for s in Dc}
    out = {"rows": rows, "stratum": stratum}
    json.dump({**out, "rho": RHO, "tau": {ch: am.TAU / RHO[ch] for ch in METHODS}}, open(CACHE, "w"))
    plot(out)
    plot_pooled(out, cellcache, mode="softmax", fname="absf_fusion_curve_pooled")
    plot_pooled(out, cellcache, mode="argmax", fname="absf_fusion_curve_pooled_argmax")
    plot_pooled(out, cellcache, mode="idw", fname="absf_fusion_curve_pooled_idw")
    plot_pooled(out, cellcache, mode="angdist", fname="absf_fusion_curve_pooled_angdist")
    plot_pooled(out, cellcache, mode="angdist_corrected", fname="absf_fusion_curve_pooled_angdist_corrected")
    plot_pooled(out, cellcache, mode="idw_corrected", fname="absf_fusion_curve_pooled_idw_corrected")
    plot_pooled(out, cellcache, mode="cossim", fname="absf_fusion_curve_pooled_cossim")


if __name__ == "__main__":
    main(sys.argv[1:])
