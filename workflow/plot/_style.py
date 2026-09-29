"""Shared minimal-seaborn plotting style for paper figures.

Conventions:
    - No grid, no background fill, despined top/right.
    - Colorblind-safe palette.
    - Large font for print readability.
    - Figures carry no embedded title (titles go in the LaTeX caption).
"""
import matplotlib.pyplot as plt
import seaborn as sns


def setup_style(font_scale: float = 1.4) -> None:
    sns.set_theme(
        style="white",
        context="talk",
        font_scale=font_scale,
        palette="colorblind",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.grid": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": True,
            "axes.titleweight": "normal",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )


def finalize(fig, output_path: str) -> None:
    fig.suptitle("")
    for ax in fig.axes:
        sns.despine(ax=ax)
    fig.savefig(output_path, bbox_inches="tight", transparent=True)
    plt.close(fig)
