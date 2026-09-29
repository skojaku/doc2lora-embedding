"""Shared barycentric-simplex drawing helpers for the #60 simplex figures.

Grid cells are integer barycentric triples (a, b, c) summing to RES (=12 -> 91
cells). Corner colours match workflow/plot/fig_simplex_fusion.py:
    A = red (#B3251B), B = blue (#457FA9), C = yellow (#F2A200).
"""
import numpy as np

RES = 12
VA = np.array([0.5, np.sqrt(3) / 2])   # A top
VB = np.array([1.0, 0.0])              # B bottom-right
VC = np.array([0.0, 0.0])              # C bottom-left
VERTS = [VA, VB, VC]


def _hex(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0


C_A, C_B, C_C = _hex("B3251B"), _hex("457FA9"), _hex("F2A200")
VCOL = np.array([C_A, C_B, C_C])


def bary_to_xy(b):
    """integer bary triple (sum=RES) or weight triple -> 2D point."""
    w = np.asarray(b, float)
    if w.sum() > 1.5:
        w = w / RES
    return w[0] * VA + w[1] * VB + w[2] * VC


def rgb_blend(b, whiten=0.15):
    """barycentric-weight RGB blend of the three corner colours."""
    w = np.asarray(b, float)
    if w.sum() > 1.5:
        w = w / RES
    col = w @ VCOL
    return tuple(np.clip(col * (1 - whiten) + whiten, 0, 1))


def draw_frame(ax, names=None, labelsize=8, lw=0.9):
    for v0, v1 in [(VA, VB), (VB, VC), (VC, VA)]:
        ax.plot([v0[0], v1[0]], [v0[1], v1[1]], "-", color="0.25", lw=lw, zorder=1)
    if names is not None:
        offs = [(0, 0.06), (0.05, -0.04), (-0.05, -0.04)]
        ha = ["center", "left", "right"]
        for v, nm, o, h, col in zip(VERTS, names, offs, ha, VCOL):
            ax.annotate(nm, v, xytext=(v[0] + o[0], v[1] + o[1]),
                        ha=h, va="center", fontsize=labelsize,
                        color=col, weight="bold", zorder=5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.32, 1.32)
    ax.set_ylim(-0.18, 1.05)


def scatter_cells(ax, barys, colors=None, values=None, cmap=None,
                  vmin=None, vmax=None, size=140):
    """draw the grid cells as hexagon markers; colours OR scalar values."""
    P = np.array([bary_to_xy(b) for b in barys])
    if values is not None:
        return ax.scatter(P[:, 0], P[:, 1], c=values, cmap=cmap, vmin=vmin,
                          vmax=vmax, s=size, marker="h", edgecolors="none", zorder=3)
    return ax.scatter(P[:, 0], P[:, 1], c=colors, s=size, marker="h",
                      edgecolors="none", zorder=3)


def method_set_grid(cellvals, sets, methods, mlabels, cmap, cbar_label, out,
                    vmin=None, vmax=None, size=70):
    """Grid of simplex triangles: methods (rows) x sets (cols), each cell shaded
    by a scalar. cellvals[(method,set)] = {bary_tuple: value}. Shared colourbar.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    allv = [v for m in methods for s in sets for v in cellvals[(m, s)].values()]
    vmin = min(allv) if vmin is None else vmin
    vmax = max(allv) if vmax is None else vmax
    nr, nc = len(methods), len(sets)
    fig, axes = plt.subplots(nr, nc, figsize=(1.7 * nc, 1.8 * nr), squeeze=False)
    sc = None
    for i, m in enumerate(methods):
        for k, s in enumerate(sets):
            ax = axes[i][k]
            d = cellvals[(m, s)]
            barys = list(d)
            sc = scatter_cells(ax, barys, values=[d[b] for b in barys], cmap=cmap,
                               vmin=vmin, vmax=vmax, size=size)
            draw_frame(ax, names=None, lw=0.6)
            if i == 0:
                ax.set_title(s, fontsize=10)
            if k == 0:
                ax.text(-0.34, 0.5, mlabels[m], transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=11, weight="bold")
    fig.subplots_adjust(right=0.9, wspace=0.05, hspace=0.12)
    cax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label(cbar_label, fontsize=11)
    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", transparent=True)
    fig.savefig(out.replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)
