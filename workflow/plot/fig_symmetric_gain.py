r"""What the shared transform g_theta does to each space, as a dot plot (replaces tab:symmetric-values).

x is the space (Doc2LoRA and ICAE first, then the five text encoders); y is the log gain
ln(score with g_theta / score without g_theta), computed per benchmark and per space and then
averaged over the panel's benchmarks. Zero (the line in every panel) is no change. One panel per
task (collaboration, next-paper and topic over Physics, Economics and Psychology; author-name
disambiguation over the five S2AND datasets) and one over all 14. Fields are never pooled inside a
benchmark; averaging happens only after the per-benchmark ratio.

Each mark is a filled circle with a 95% two-level bootstrap interval: every replicate resamples the
panel's benchmarks with replacement and redraws each raw and each adapted score from its own
per-benchmark bootstrap, then recomputes the ratio. The summaries keep only lo/hi, so a score is
redrawn from a normal with s.d. (hi - lo) / 3.92.

Method keys: Doc2LoRA raw `gene`, adapted `genkron` (the reported transform, the Doc2LoRA row of
tab:similarity); ICAE `icae` / `icae_genkron`; every text encoder `<enc>` / `<enc>_kron_gc`.
SPECTER2 on the field benchmarks and SPECTER (v1) on S2AND are drawn as one space.

Dual-mode: Snakemake or CLI.
  python workflow/plot/fig_symmetric_gain.py <gcb_summary.csv> <gcb_s2and_summary.csv> <out.pdf> [out.png]
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = "#0b0b0b"
GUIDE = "#8a8985"     # the zero (no change) line
N_BOOT = 4000
SEED = 0

FIELDS = ["aps", "economics", "psychology"]
S2AND = ["zbmath", "qian", "arnetminer", "pubmed", "kisti"]
# (tick label, raw key, adapted key); S2AND keys differ only for SPECTER
SPACES = [("D2L", "gene", "genkron"),
          ("ICAE", "icae", "icae_genkron"),
          ("SBERT", "sbert", "sbert_kron_gc"),
          ("EmbGemma", "embeddinggemma", "embeddinggemma_kron_gc"),
          ("Instructor", "instructor", "instructor_kron_gc"),
          ("SPECTER", "specter2", "specter2_kron_gc"),
          ("GTE", "gte", "gte_kron_gc")]
S2AND_ALIAS = {"specter2": "specter", "specter2_kron_gc": "specter_kron_gc"}

BENCHMARKS = ([(f, t) for t in ("collab", "np", "topic") for f in FIELDS]
              + [(ds, "name_disambig") for ds in S2AND])
PANELS = [("Collaboration", "collab"), ("Next paper", "np"), ("Topic", "topic"),
          ("Disambiguation", "name_disambig"), ("All", None)]


def short(v, dec):
    """-0.73 -> '\u2212.73', 0 -> '0'."""
    if abs(v) < 1e-9:
        return "0"
    return f"{v:.{dec}f}".replace("0.", ".").replace("-", "\u2212")


def matrices(field_csv, s2and_csv):
    """(value, s.d.) arrays shaped space x benchmark, for raw and adapted."""
    A = pd.concat([pd.read_csv(field_csv), pd.read_csv(s2and_csv)], ignore_index=True)
    look = {(r.field, r.task, r.method): (r.value, (r.hi - r.lo) / 3.92)
            for r in A.itertuples()}
    shape = (len(SPACES), len(BENCHMARKS))
    X = {k: np.full(shape, np.nan) for k in ("raw", "adapted")}
    S = {k: np.zeros(shape) for k in ("raw", "adapted")}
    for j, (field, task) in enumerate(BENCHMARKS):
        for i, (_, raw, adapted) in enumerate(SPACES):
            for kind, key in (("raw", raw), ("adapted", adapted)):
                if task == "name_disambig":
                    key = S2AND_ALIAS.get(key, key)
                X[kind][i, j], S[kind][i, j] = look[(field, task, key)]
    return X, S


def summarise(X, S, cols, rng):
    point = np.log(X["adapted"][:, cols] / X["raw"][:, cols]).mean(axis=1)
    nb = len(cols)
    draws = np.empty((N_BOOT, len(SPACES)))
    for r in range(N_BOOT):
        idx = [cols[k] for k in rng.integers(0, nb, nb)]
        raw = X["raw"][:, idx] + rng.normal(size=(len(SPACES), nb)) * S["raw"][:, idx]
        ad = X["adapted"][:, idx] + rng.normal(size=(len(SPACES), nb)) * S["adapted"][:, idx]
        draws[r] = np.log(ad / raw).mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return point, lo, hi


def build(field_csv, s2and_csv, out_pdf, png=None):
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 7,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    X, S = matrices(field_csv, s2and_csv)
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, len(PANELS), figsize=(5.5, 2.3), sharey=True)
    x = np.arange(len(SPACES))
    ends = []
    for ax, (title, task) in zip(axes, PANELS):
        cols = [j for j, (_, t) in enumerate(BENCHMARKS) if task is None or t == task]
        point, lo, hi = summarise(X, S, cols, rng)
        ends += [lo.min(), hi.max()]
        print(f"\n[{title}] n={len(cols)}  mean ln(adapted / raw)")
        for i, (lbl, _, _) in enumerate(SPACES):
            print(f"  {lbl:<11} {point[i]:+.3f} [{lo[i]:+.3f}, {hi[i]:+.3f}]")
        ax.axhline(0, color=GUIDE, lw=0.6, zorder=0)
        ax.vlines(x, lo, hi, color=INK, lw=0.8, zorder=2)
        ax.scatter(x, point, s=14, facecolor=INK, edgecolor=INK, lw=0.7, zorder=3)
        n = len(cols)
        ax.set_title(f"{title}\n{n} benchmarks", fontsize=7, color=INK, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels([s[0] for s in SPACES], rotation=90, fontsize=7)
        ax.set_xlim(-0.6, len(SPACES) - 0.4)
        ax.tick_params(axis="both", length=2.5, pad=2, labelsize=7)
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False)
        ax.spines["left"].set_visible(False)
    # one linear axis wide enough for every interval, so no mark falls off it
    axes[0].set_ylim(min(-0.1, min(ends) - 0.03), max(ends) + 0.05)
    axes[0].yaxis.set_major_locator(plt.MultipleLocator(0.2))
    axes[0].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: short(v, 1)))
    axes[0].set_ylabel("ln(with $g_\\theta$ / without)", fontsize=7)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(out_pdf, bbox_inches="tight")
    if png:
        fig.savefig(png, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig_symmetric_gain] -> {out_pdf}")


if __name__ == "__main__":
    if "snakemake" in globals():
        build(snakemake.input["fields"], snakemake.input["s2and"],  # noqa: F821
              snakemake.output["pdf"])  # noqa: F821
    else:
        build(*sys.argv[1:5])
