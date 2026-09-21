"""Overlay the pooled fusion-response curve across embedding backends (fixed tau0, no opt).
Small multiples: one panel per embedding model, 3 method lines w/ bootstrap 95% CI.
Consistent shape across panels => picture is embedding-model-independent.

  python plot_embcurves.py                      # all embcurve_*.json present
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import absf_metrics as am
import simplex_common as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = am.METHODS
ORDER = ["mpnet", "gtr", "specter2", "gemma"]
NICE = {"mpnet": "all-mpnet (SBERT)", "gtr": "GTR-T5", "specter2": "SPECTER2", "gemma": "EmbeddingGemma"}


def boot_ci(vals, B=1000, seed=0):
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = np.asarray(vals)[rng.integers(0, len(vals), size=(B, len(vals)))].mean(1)
    return float(np.mean(vals)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    sns.set_theme(style="white", palette="colorblind", font_scale=1.15)
    files = {os.path.basename(f)[9:-5]: f for f in glob.glob(os.path.join(sc.RESULTS, "embcurve_*.json"))}
    backs = [b for b in ORDER if b in files] + [b for b in files if b not in ORDER]
    bins = np.linspace(0, 1, 21); ctr = 0.5 * (bins[1:] + bins[:-1])
    n = len(backs)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.5), squeeze=False)
    for k, bk in enumerate(backs):
        ax = axes[0, k]; ax.grid(False)
        d = json.load(open(files[bk]))["pts"]
        ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1)
        for ch in METHODS:
            xs = np.array(d[ch]["x"]); ys = np.array(d[ch]["y"])
            m, lo, hi = [], [], []
            for i in range(len(bins) - 1):
                sel = ys[(xs >= bins[i]) & (xs < bins[i + 1] + 1e-9)]
                a, b, c = boot_ci(sel)
                m.append(a); lo.append(b); hi.append(c)
            z = 6 if ch == "doc2lora" else 3
            ax.fill_between(ctr, lo, hi, color=am.MCOL[ch], alpha=0.2, zorder=z - 1)
            ax.plot(ctr, m, "-o", color=am.MCOL[ch], ms=3.5,
                    lw=3.0 if ch == "doc2lora" else 1.8, label=am.LAB[ch], zorder=z)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel("ideal weight")
        if k == 0:
            ax.set_ylabel("measured weight"); ax.legend(fontsize=12, frameon=False, loc="upper left")
        ax.set_title(NICE.get(bk, bk), fontsize=13)
        ax.tick_params(labelsize=12)
        sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figs", "absf_embcurves.pdf"), bbox_inches="tight", dpi=300, transparent=True)
    plt.close(fig)
    print("saved figs/absf_embcurves.pdf  (backends: %s)" % ", ".join(backs))


if __name__ == "__main__":
    main()
