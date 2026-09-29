"""Palette and helpers shared by the Figure 2 family (fig_pacs_clustering.py,
the table script). No side effects on import.

Three colours only, as in the teams4industry paper: grey is the base, orange
marks Doc2LoRA (the pattern the figure is about) and blue is the opposite pole
of a diverging scale. Text wears ink, never a series colour.
"""
import numpy as np
import seaborn as sns

INK, INK2 = "#0b0b0b", "#52514e"                # text tokens
SURFACE = "#ffffff"
ORANGE = "#eb6834"                              # categorical slot 2: Doc2LoRA
BLUE = "#2a78d6"                                # categorical slot 1
GREY_MARK, GREY_MARK2 = "#6f6e6a", "#a8a7a1"    # baseline marks (dark / light)
GREY_FILL, GREY_LIGHT = "#dedcd7", "#f0efec"    # baseline wash / neutral ramp start
# the diverging scale for a quantity with a meaningful middle (win rate, 0.5 = tie):
# blue (low) -> neutral grey -> orange (high). Grey is the base everywhere; blue
# and orange are spent only where a value carries meaning.
DIVERGING = [BLUE, GREY_LIGHT, ORANGE]

# the four arms of the mixing panels: orange for Doc2LoRA, three greys (dark to
# light) for the baselines -- luminance is the only channel, so end labels name them
LINE_COLOR = {"doc2lora": ORANGE, "icae": "#3d3c39", "incontext": "#8b8a85", "t2l": "#c2c1bb"}
# short names: D2L is what the manuscript's \doctolora macro prints; In-ctx is
# spelled out in the caption
MLAB = {"doc2lora": "D2L", "icae": "ICAE", "incontext": "In-ctx",
        "keyllm": "KeyLLM", "vec2text": "vec2text", "t2l": "T2L"}


def set_style():
    sns.set_theme(style="white", rc={"pdf.fonttype": 42, "ps.fonttype": 42,
                                     "font.family": "DejaVu Sans",
                                     "text.color": INK, "axes.labelcolor": INK,
                                     "xtick.color": INK2, "ytick.color": INK2,
                                     "axes.edgecolor": INK2})


def panel_letter(ax, letter, fontsize, dx=-0.10, dy=1.02):
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", ha="left", va="bottom", clip_on=False)


def boot_ci(vals, B=2000, seed=0):
    """Mean and 95 % bootstrap interval of the mean (percentile, B resamples)."""
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(B, len(vals)))
    means = np.asarray(vals)[idx].mean(1)
    return float(np.mean(vals)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fmt(x, nd=2):
    """0.596 -> '.60' (leading zero dropped, as in the paper's tables); 1.00 -> '1.0'."""
    if x >= 1 - 0.5 * 10 ** (-nd):
        return "1.0"
    s = f"{x:.{nd}f}"
    return s[1:] if s.startswith("0") else s
