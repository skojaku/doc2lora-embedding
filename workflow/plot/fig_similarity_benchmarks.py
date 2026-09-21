r"""Standardised similarity benchmarks, one panel per task (replaces tab:similarity).

The 14 benchmarks sit at wildly different levels (macro-F1 .24 on Economics
topics, AUC .96 on Physics next-paper), so none of them may be averaged raw.
Each benchmark is centred on the mean of the methods compared on it, and a
method's mark in a panel is its mean centred score over that task's
fields/datasets, with a last panel over all 14. The unit is a point of that
benchmark's own metric and zero -- the line in every panel -- is the mean method
there. The spread is left alone (no division by a s.d.), so a benchmark on which
the methods disagree more counts for more in the average. Because points of AUC
and of macro-F1 cannot be averaged with each other, the panel over all 14 is a
mean RANK instead (1 = best), drawn on a reversed axis so that better still
means further right. Absolute per-benchmark scores are in the appendix table
(figs/similarity_benchmarks.tex).

The interval is a two-level bootstrap: it resamples the panel's benchmarks (the
benchmark-to-benchmark variation, which dominates) and redraws every score from
its own bootstrap distribution (the measurement noise). Methods are drawn
independently, as in the per-benchmark bootstrap, so the interval is that
method's own and comparing two of them is conservative.

Marker shape names the method and every row carries its own label on the left,
so identity never rests on colour and the figure still reads in black and white.
Every method but raw \doctolora wears a hue (the author's call), drawn from the
dataviz screen palette with green left out: orange for \doctolora with g_theta,
blue for SBERT, red for GTE, violet for EmbeddingGemma, magenta for Instructor,
gold for SPECTER2, and neutral ink for raw \doctolora, which carries no claim.
The set passes the dataviz validator as an adjacent-pair categorical palette on
white (worst adjacent CVD dE 13.6, normal 20.9, all six above 3:1 contrast); six
hues cannot clear the all-pairs floors, which is why shape and row label carry
the identity. Text and axes wear ink, never a series colour.

Dual-mode: Snakemake or CLI.
  python workflow/plot/fig_similarity_benchmarks.py <summary.csv> <out.pdf>
"""
import csv
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

INK = "#0b0b0b"      # all text, axes and ticks
MARK = "#6f6e6a"     # a method with no claim on it
ORANGE = "#eb6834"   # doc2lora with g_theta
BLUE = "#2a78d6"     # SBERT
RED = "#d43d3c"      # GTE (sits directly under SBERT, the encoder it is read against)
VIOLET = "#4a3aa7"   # EmbeddingGemma
MAGENTA = "#d0528a"  # Instructor
GOLD = "#b8860b"     # SPECTER2
AXIS = "#8a8985"     # the zero reference line
BAND = "#eeede9"     # alternating row band

N_BOOT = 4000
SEED = 0


def tick_label(dec):
    """A formatter with `dec` decimals: .05 -> '.05', -.05 -> '-.05', 0 -> '0'."""
    def fmt(x, _pos=None):
        if abs(x) < 1e-9:
            return "0"
        return f"{x:.{dec}f}".replace("0.", ".")
    return fmt


def set_style():
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.facecolor": "white", "figure.facecolor": "white",
    })


MAIN_ENC = "qwen"
# doc2lora runs are keyed by the encoder for the field tasks and by "gene" for
# the S2AND disambiguation datasets.
D2L_BASE = {"name_disambig": "gene"}
# SPECTER2 is only available as SPECTER on the disambiguation datasets.
KEY_ALIAS = {("name_disambig", "specter2"): "specter"}

# row order, top to bottom, as the author set it
ROWS = [
    ("D2L + $g_\\theta$", ("d2l", "_genkron")),
    ("D2L raw", ("d2l", "")),
    ("SBERT", ("enc", "sbert")),
    ("GTE", ("enc", "gte")),
    ("Emb.Gemma", ("enc", "embeddinggemma")),
    ("Instructor", ("enc", "instructor")),
    ("SPECTER2", ("enc", "specter2")),
]

# One tier per task; inside a tier one panel per field/dataset, then an "All"
# panel averaging that tier's benchmarks. A tier's metric is the same
# throughout, so its panels and the All panel share the metric's own points.
FIELDS = [("aps", "Physics"), ("economics", "Economics"),
          ("psychology", "Psychology")]
TIERS = [
    ("Next-paper (AUC)", "next_paper", FIELDS),
    ("Topic (macro-F1)", "topic", FIELDS),
    ("Collaboration (AUC)", "collab", FIELDS),
    ("Disambiguation (B³ F1)", "name_disambig",
     [("zbmath", "zbMATH"), ("qian", "QIAN"), ("arnetminer", "ArnetMiner"),
      ("pubmed", "PubMed"), ("kisti", "KISTI")]),
]
BENCHMARKS = [(bench, g) for _, bench, groups in TIERS for g, _ in groups]

# tier -> [(panel title, benchmark columns)], the last one being "All"
PANELS = []
_at = 0
for _t, _b, _gs in TIERS:
    cols = list(range(_at, _at + len(_gs)))
    PANELS.append([(lab, [c]) for (_, lab), c in zip(_gs, cols)] + [("All", cols)])
    _at += len(_gs)

# One hue per method in row order, raw \doctolora excepted: it makes no claim,
# so it stays neutral. Validated as an adjacent-pair categorical set with the
# dataviz validator on white (worst adjacent CVD dE 13.6, normal 20.9).
ROW_COLOR = {
    "D2L + $g_\\theta$": ORANGE, "SBERT": BLUE, "GTE": RED,
    "Emb.Gemma": VIOLET, "Instructor": MAGENTA, "SPECTER2": GOLD,
}

# a shape per method, so the figure survives a black-and-white print. No
# triangles (the author's call). Scatter sizes are areas, so the shapes that
# waste their bounding box get more of it.
ROW_MARKER = {
    "D2L + $g_\\theta$": "o", "D2L raw": "D", "SBERT": "s", "GTE": "h",
    "Emb.Gemma": "P", "Instructor": "X", "SPECTER2": "*",
}
MARKER_AREA = {"o": 13, "D": 16, "s": 12, "P": 19, "X": 17, "*": 30, "h": 16}


def load(csv_path):
    """key -> (mean, s.d.) of the per-benchmark bootstrap distribution."""
    with open(csv_path) as f:
        return {(r["benchmark"], r["group"], r["enc"], r["method"]):
                (float(r["mean"]), float(r["std"])) for r in csv.DictReader(f)}


def key_for(row, bench, enc):
    kind, key = row
    if kind == "d2l":
        return D2L_BASE.get(bench, enc) + key
    return KEY_ALIAS.get((bench, key), key)


def present_rows(D, enc):
    """Drop a method with no data anywhere, so the figure never claims a run."""
    keys = {m for (_, _, _, m) in D}
    return [r for r in ROWS
            if any(key_for(r[1], b, enc) in keys for b, _ in BENCHMARKS)]


def matrices(D, rows, enc):
    """(scores, s.d.) as method x benchmark, NaN where a method was not run."""
    X = np.full((len(rows), len(BENCHMARKS)), np.nan)
    S = np.zeros_like(X)
    for j, (bench, group) in enumerate(BENCHMARKS):
        for i, (_, key) in enumerate(rows):
            v = D.get((bench, group, enc, key_for(key, bench, enc)))
            if v is not None:
                X[i, j], S[i, j] = v
    return X, S


def centre(X):
    """Per benchmark: subtract the mean over the methods compared on it."""
    return X - np.nanmean(X, axis=0)


def summarise(X, S, cols, rng):
    """Mean deviation from the benchmark's mean method, with a bootstrap CI.

    The CI is two-level: it resamples the panel's benchmarks (nothing to
    resample when a panel is a single benchmark) and redraws every score from
    its own per-benchmark bootstrap distribution.
    """
    point = np.nanmean(centre(X)[:, cols], axis=1)
    nb = len(cols)
    draws = np.empty((N_BOOT, X.shape[0]))
    for r in range(N_BOOT):
        idx = [cols[k] for k in rng.integers(0, nb, nb)]    # resample benchmarks
        noise = rng.normal(0.0, 1.0, (X.shape[0], nb))      # measurement noise
        draws[r] = np.nanmean(centre(X[:, idx] + noise * S[:, idx]), axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return point, lo, hi


def build(csv_path, out_pdf):
    set_style()
    D = load(csv_path)
    rows = present_rows(D, MAIN_ENC)
    X, S = matrices(D, rows, MAIN_ENC)
    n = len(rows)

    rng = np.random.default_rng(SEED)
    tiers = [[(title, summarise(X, S, cols, rng)) for title, cols in panels]
             for panels in PANELS]

    # Height is set by the row pitch, not by taste: seven rows per tier each
    # carry their own label on the left, so a tier needs 7 x ~8pt of vertical
    # room or the labels collide. 4.55in leaves ~8pt per row after the tier
    # gaps, which is what a 6pt label wants.
    fig = plt.figure(figsize=(5.5, 4.55))
    outer = fig.add_gridspec(len(TIERS), 1, left=0.155, right=0.995,
                             top=0.955, bottom=0.03, hspace=0.42)

    for ti, (panels, (task, _, _)) in enumerate(zip(tiers, TIERS)):
        inner = outer[ti].subgridspec(1, len(panels), wspace=0.12)
        axes = [fig.add_subplot(inner[pi]) for pi in range(len(panels))]
        # the field columns repeat down the first three tiers: title them once
        titled = ti == 0 or ti == len(TIERS) - 1

        for pi, (ax, (title, (point, lo, hi))) in enumerate(zip(axes, panels)):
            for i in range(n):
                if i % 2 == 0:
                    ax.axhspan(i - 0.5, i + 0.5, color=BAND, alpha=0.55, lw=0,
                               zorder=0)
            ax.axvline(0, color=AXIS, lw=0.7, zorder=1)     # the mean method
            for i in range(n):
                color = ROW_COLOR.get(rows[i][0], MARK)
                marker = ROW_MARKER.get(rows[i][0], "o")
                ax.plot([lo[i], hi[i]], [i, i], color=color, lw=0.8, alpha=0.5,
                        solid_capstyle="butt", zorder=2)
                # the mark wears a white ring so it never merges with its whisker
                ax.scatter([point[i]], [i], s=MARKER_AREA[marker], marker=marker,
                           zorder=3, facecolors=color, edgecolors="white",
                           linewidths=0.6)

            if titled:
                ax.set_title(title, fontsize=6.3, color=INK, pad=2.5)
            # each panel is scaled to its own data; zero is always inside it,
            # because the marks are deviations from that benchmark's mean
            span = hi.max() - lo.min()
            ax.set_xlim(lo.min() - 0.08 * span, hi.max() + 0.08 * span)
            ax.set_ylim(n - 0.5, -0.5)
            ax.set_yticks(range(n))
            ax.set_yticklabels([lab for lab, _ in rows] if pi == 0 else [],
                               fontsize=6.0, color=INK)
            ax.tick_params(axis="y", length=0)
            ax.tick_params(axis="x", labelsize=5.5, length=1.8, width=0.5,
                           pad=1.2, labelcolor=INK)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3, prune="both"))
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(tick_label(2 if span >= 0.04 else 3)))
            ax.spines["bottom"].set_linewidth(0.6)

        # the header clears the column titles where a tier carries them, and is
        # centred over the panels of its tier
        first, last = axes[0].get_position(), axes[-1].get_position()
        dy = (0.135 if titled else 0.035) / fig.get_figheight()
        # no bold anywhere: the tier header is set apart by size alone
        fig.text(0.5 * (first.x0 + last.x1), first.y1 + dy, task, fontsize=7.4,
                 color=INK, ha="center", va="baseline")

    # tight with a hair of padding: the PDF then lands at ~\textwidth (5.5 in)
    # so \includegraphics[width=\textwidth] renders the type at its true size.
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01, dpi=300,
                transparent=True)
    plt.close(fig)
    print(f"[fig_similarity_benchmarks] -> {out_pdf}")


if __name__ == "__main__":
    if "snakemake" in sys.modules:
        build(snakemake.input["summary"], snakemake.output["pdf"])  # noqa: F821
    else:
        build(sys.argv[1], sys.argv[2])
