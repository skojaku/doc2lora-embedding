"""Figure 2 (figs/pacs-clustering.pdf): cluster labels and mixtures in Doc2LoRA.

Six panels in three columns, drawn from cached numbers only (no GPU, no corpus,
no judge calls). Each input is produced by a workflow rule:

  column 1 -- how good the labels are
  (a) fuzzy token-set overlap between each method's decoded label and the
      official PACS name, one point per node (28), KDE behind, mean + 95 % CI.
        <- label_eval_metric1.json      (baseline_trees.smk:bt_label_eval_metric1)
  (b) five-judge round robin: the row method's win rate against the column
      method (ties 0.5), row mean at the right.
        <- label_eval_metric4.json      (baseline_trees.smk:bt_label_eval_metric4;
                                          passed as a param so the judge panel
                                          stays out of the `paper` DAG)
  column 2 -- why: the length of the averaged vector (x = vector length in both)
  (c) PACS node centroids: one row per hierarchy level, x = ||v||_2 (breadth),
      circle size = papers; as many nodes as the room allows (14) carry the
      decoded label in orange and, where it names another field, the official
      PACS name in grey (pack_callouts: labels above their dot on vertical leaders).
        <- length_vs_breadth.csv        (baseline_trees.smk:bt_length_dial)
           qwen_fullrank_field23.json   (baseline_trees.smk:bt_decode_fullrank)
  (d) Wikipedia abstraction walk: one row per seed document, x = the scale
      alpha applied to its embedding (||v|| = alpha ||v_doc||), one label per
      run of equal decoded labels, on the run's outermost step (its largest
      alpha); a label that finds no room is dropped. The axis is the scale,
      not the length: a walk step is the full 288-row gene (||v_doc|| =
      sqrt(288) = 17.0 for every seed) while (c) reads the rank-pooled,
      corpus-averaged gene (||.|| ~ 3), so the two lengths do not compare.
        <- abstraction_walk.json        (abstraction_walk.smk:aw_doc2lora)
  column 3 -- mixing two documents (L1 stratum, 50 near pairs)
  (e) actual against ideal mixing, bootstrap 95 % CI; T2L at 3 points only.
        <- pair_axis_metrics_pairaxis.json    (pair_axis_metrics.py)
           pair_axis_t2l_<space>_pairaxis.json (fig2_t2l_edge_metrics.py)
  (f) verbatim copy rate against ideal mixing, same arms.
        <- pair_axis_copyrate_pairaxis_L1.json (pair_axis_copyrate.py) + the T2L cache

The decoded example along one edge (the former colorband panel) is Tab.
mixing-decode (workflow/scripts/fig2_mixing_table.py -> figs/mixing_decode.tex).

Colour policy (_fig2_palette.py): grey is the base; orange marks Doc2LoRA and
the winning side of the win-rate scale, blue its losing side; text in ink.
Method names are short (D2L = Doc2LoRA, In-ctx = in-context reader); the
caption spells them out.

Dual mode: driven by Snakemake (workflow/rules/fig2_pacs_clustering.smk) or
run from the repo root with the default paths:

    python workflow/plot/fig_pacs_clustering.py [--out figs/pacs-clustering.pdf] [--png preview.png]
"""
import argparse
import itertools
import json
import os
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mc
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _fig2_palette import (DIVERGING, GREY_FILL, GREY_LIGHT, GREY_MARK, GREY_MARK2, INK,  # noqa: E402
                           INK2, LINE_COLOR, MLAB, ORANGE, SURFACE, boot_ci, fmt, set_style)
from _fig2_palette import panel_letter as _panel_letter  # noqa: E402

# the label-quality arms, in display order; T2L is its best space (chosen below)
EVAL_ARMS = ["doc2lora", "icae", "incontext", "keyllm", "vec2text", "t2l"]
MIX_ARMS = ["doc2lora", "icae", "incontext", "t2l"]

# 16.4 x 12 in, printed at \linewidth (5.5 in), i.e. 0.34x: axis text ~7-8 pt,
# annotations ~5-6 pt in print.
FS_TICK, FS_LABEL, FS_LEGEND, FS_TEXT, FS_PANEL = 20, 24, 20, 17, 32


def panel_letter(ax, letter, dx=-0.10, dy=1.02):
    _panel_letter(ax, letter, FS_PANEL, dx=dx, dy=dy)


def _t2l_space(m1):
    """The paper reports T2L as its best of three spaces; pick it by fuzzy mean."""
    cands = {k: v["fuzzy_mean"] for k, v in m1["summary"].items() if k.startswith("t2l")}
    if not cands:
        raise KeyError("no t2l_* arm in the fuzzy results")
    return max(cands, key=cands.get)


def _units_per_inch(ax, fig):
    """Data units per inch along x and y for the axes' current box and limits."""
    bb = ax.get_position()
    w_in, h_in = bb.width * fig.get_figwidth(), bb.height * fig.get_figheight()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return (x1 - x0) / w_in, (y1 - y0) / h_in


def _text_w(ax, s, fs, bold=False, italic=False):
    """Rendered width of `s` in inches, measured with the figure's renderer."""
    t = ax.text(0, 0, s, fontsize=fs, fontweight="bold" if bold else "normal",
                style="italic" if italic else "normal")
    w = t.get_window_extent(renderer=ax.figure.canvas.get_renderer()).width / ax.figure.dpi
    t.remove()
    return w


# word-level abbreviations for the annotation text; the caption spells them out
ABBREV = {"physics": "phys.", "Physics": "Phys.", "mechanics": "mech.", "Mechanics": "Mech.",
          "Statistical": "Stat.", "Condensed": "Cond.", "chromodynamics": "chromodyn.",
          "Elementary": "Elem.", "Interdisciplinary": "Interdisc.", "Fractional": "Frac.",
          "Semiconductor": "Semicond.", "structure": "struct.", "characteristics": "charact.",
          "metabolism": "metab.", "and": "&", "Effect": "eff.", "effect": "eff.",
          "respiration": "resp.", "engineering": "eng."}
ABBREV_PHRASES = [("Light-emitting diode", "LED")]


def abbrev(text):
    for a, b in ABBREV_PHRASES:
        text = text.replace(a, b)
    return " ".join(ABBREV.get(w, w) for w in text.split())


def _tier_positions(todo, obstacles, forbidden, xmin, xmax, gap, inset):
    """1-D placement of the labels `todo` (sorted by x, each with "x", "w") in
    one tier: every span contains its own x (the leader meets the text at
    least `inset` inside its ends), spans keep `gap` from each other and from
    the fixed `obstacles`, and no span covers a point of `forbidden` (leaders
    that pass through this tier). Returns the left edges, or None."""
    inf = float("inf")
    windows = []
    for it in todo:
        w = it["w"]
        ins = min(inset, w / 2)
        fl = max([f for f in forbidden if f < it["x"]], default=-inf) + gap / 2
        fr = min([f for f in forbidden if f > it["x"]], default=inf) - gap / 2
        lo = max(xmin, it["x"] - w + ins, fl)
        hi = min(xmax - w, it["x"] - ins, fr - w)
        if lo > hi + 1e-9:
            return None
        windows.append((lo, hi))

    def clear(s, w):
        moved = True
        while moved:
            moved = False
            for a, b in obstacles:
                if a - w - gap < s < b + gap:
                    s, moved = b + gap, True
        return s

    # leftmost feasible packing (decides feasibility) ...
    pos, prev_end = [], -inf
    for it, (lo, hi) in zip(todo, windows):
        s0 = clear(max(lo, prev_end + gap), it["w"])
        if s0 > hi + 1e-9:
            return None
        pos.append(s0)
        prev_end = s0 + it["w"]
    # ... then relax every label toward the centre of its dot within the slack
    for _ in range(30):
        new = [min(max(it["x"] - it["w"] / 2, lo), hi) for it, (lo, hi) in zip(todo, windows)]
        for k in range(len(new)):                                     # forward: keep the gaps
            if k and new[k] < new[k - 1] + todo[k - 1]["w"] + gap:
                new[k] = new[k - 1] + todo[k - 1]["w"] + gap
            new[k] = clear(new[k], todo[k]["w"])
        ok = all(lo - 1e-9 <= v <= hi + 1e-9 for v, (lo, hi) in zip(new, windows))
        if ok:
            pos = new
        for k in range(len(new) - 2, -1, -1):                         # backward
            if new[k] + todo[k]["w"] + gap > new[k + 1]:
                new[k] = new[k + 1] - gap - todo[k]["w"]
        ok = all(lo - 1e-9 <= v <= hi + 1e-9 for v, (lo, hi) in zip(new, windows)) and \
            all(not (a - it["w"] - gap < v < b + gap) for v, it in zip(new, todo) for a, b in obstacles)
        if ok:
            pos = new
            break
    return pos


def pack_callouts(items, xmin, xmax, ux, max_tiers, gap_in=0.12, inset_in=0.06):
    """Lay out labels that sit ABOVE their dot on a straight vertical leader.

    Each item: {"x": leader position, "variants": [(width, height_in_tiers),
    ...], "prio": smaller = admitted first}. A label may slide left or right
    as long as the leader still meets the text (at least `inset_in` inside
    its ends), so a wide label can use empty room on either side. Rules that
    keep every pairing unambiguous: labels in one tier stay `gap_in` apart, a
    label never covers the leader of a label above it, and a leader never
    crosses a label below it. Items are admitted in priority order while the
    admitted set still has an arrangement within `max_tiers` (every
    assignment of items to tiers and text variants is tried, the cheapest --
    least displaced from centred, lowest -- wins); the rest are dropped.
    Returns the placed items with "s" (left edge), "w", "h" and "tier".
    """
    gap, inset = gap_in * ux, inset_in * ux

    def arrange(group):
        opts = [[(vi, k) for vi, (w, h) in enumerate(it["variants"]) for k in range(max_tiers - h + 1)]
                for it in group]
        best = None
        for combo in itertools.product(*opts):
            cost = 0.0
            spec = []
            for it, (vi, k) in zip(group, combo):
                w, h = it["variants"][vi]
                spec.append({**it, "w": w, "h": h, "tier": k})
                cost += 0.5 * ux * k + (0.4 * ux if h > 1 else 0.0)
            if best is not None and cost >= best[0]:
                continue
            fixed = {}                                        # tier -> spans of taller items from below
            placed, feasible = [], True
            for k in range(max_tiers):
                todo = sorted([q for q in spec if q["tier"] == k], key=lambda q: q["x"])
                forbidden = [q["x"] for q in spec if q["tier"] > k]
                pos = _tier_positions(todo, fixed.get(k, []), forbidden, xmin, xmax, gap, inset) if todo else []
                if pos is None:
                    feasible = False
                    break
                for q, s0 in zip(todo, pos):
                    q = {**q, "s": s0}
                    placed.append(q)
                    cost += abs(s0 + q["w"] / 2 - q["x"])
                    for kk in range(k + 1, k + q["h"]):
                        fixed.setdefault(kk, []).append((s0, s0 + q["w"]))
                if best is not None and cost >= best[0]:
                    feasible = False
                    break
            if feasible and (best is None or cost < best[0]):
                best = (cost, placed)
        return best

    accepted, result = [], []
    for it in sorted(items, key=lambda d: d["prio"]):
        r = arrange(accepted + [it])
        if r is not None:
            accepted, result = accepted + [it], r[1]
    return result


def draw_leader(ax, x, y, y_top, uy):
    """Vertical leader from just above the dot at (x, y) up to y_top."""
    ax.plot([x, x], [y + 0.05 * uy, y_top], color=GREY_MARK, lw=0.9, zorder=2)


# ── (a) fuzzy overlap per node ────────────────────────────────────────────
def _kde(x, grid, lo=0.0, hi=1.0):
    """Gaussian KDE with reflection at both bounds (scores live on [0, 1])."""
    x = np.asarray(x, float)
    n = len(x)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    spread = min(x.std(ddof=1), iqr / 1.34) if iqr > 0 else x.std(ddof=1)
    bw = max(0.6 * spread * n ** (-0.2), 0.03)      # 0.6x Silverman: sharper than the default
    pts = np.concatenate([x, 2 * lo - x, 2 * hi - x])
    z = (grid[:, None] - pts[None, :]) / bw
    return np.exp(-0.5 * z ** 2).sum(1) / (n * bw * np.sqrt(2 * np.pi))


def panel_fuzzy(ax, m1_json, arm_keys):
    d = json.load(open(m1_json))
    grid = np.linspace(0, 1, 240)
    rng = np.random.default_rng(0)
    for i, arm in enumerate(EVAL_ARMS):
        key = arm_keys[arm]
        x = np.array([n["scores"][key]["fuzzy"] for n in d["per_node"]])
        is_ours = arm == "doc2lora"
        col = ORANGE if is_ours else GREY_MARK
        dens = _kde(x, grid)
        dens = dens / dens.max() * 0.40
        ax.fill_betweenx(grid, i - dens, i + dens, color=ORANGE if is_ours else GREY_FILL,
                         alpha=0.22 if is_ours else 1.0, lw=0, zorder=1)
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(x)), x, s=40, color=col,
                   edgecolors=SURFACE, linewidths=0.8, zorder=3)
        mean, lo, hi = boot_ci(x)
        ax.plot([i, i], [lo, hi], color=INK, lw=2.4, solid_capstyle="round", zorder=4)
        ax.plot([i], [mean], marker="D", ms=10, color=INK, mec=SURFACE, mew=1.2, ls="none", zorder=5)
        ax.text(i, 1.04, fmt(mean), ha="center", va="bottom", fontsize=FS_TICK - 1, color=INK2)
    ax.set_xticks(range(len(EVAL_ARMS)))
    ax.set_xticklabels([MLAB[a] for a in EVAL_ARMS], rotation=30, ha="right", rotation_mode="anchor",
                       fontsize=FS_TICK)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", ".25", ".5", ".75", "1"], fontsize=FS_TICK)
    ax.set_xlim(-0.6, len(EVAL_ARMS) - 0.4)
    ax.set_ylim(-0.02, 1.14)
    ax.set_ylabel("Fuzzy overlap", fontsize=FS_LABEL)
    ax.tick_params(axis="x", length=0)
    sns.despine(ax=ax)


# ── (b) judge round robin ─────────────────────────────────────────────────
WIN_CMAP = LinearSegmentedColormap.from_list("blue_grey_orange", DIVERGING)   # 0 = loses, .5 = tie, 1 = wins


def panel_judge(ax, fig, m4_json, arm_keys):
    d = json.load(open(m4_json))
    h = d["head_to_head"]
    keys = [arm_keys[a] for a in EVAL_ARMS]
    n = len(keys)
    M = np.full((n, n), np.nan)
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if i == j:
                continue
            rec = h.get(f"{a}|{b}") or h.get(f"{b}|{a}")
            if rec is None:
                raise KeyError(f"no head-to-head record for {a} vs {b} in {m4_json}")
            M[i, j] = rec[f"rate_{a}"]
    row_mean = np.nanmean(M, axis=1)

    GAP = 0.4                      # between the matrix and the row-mean column
    x_mean = n + GAP
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(patches.Rectangle((j, i), 1, 1, facecolor=SURFACE, edgecolor=SURFACE, lw=2))
                ax.text(j + 0.5, i + 0.5, "–", ha="center", va="center", fontsize=FS_TICK, color=GREY_MARK2)
                continue
            ax.add_patch(patches.Rectangle((j, i), 1, 1, facecolor=WIN_CMAP(M[i, j]),
                                           edgecolor=SURFACE, lw=2))
            ax.text(j + 0.5, i + 0.5, fmt(M[i, j]), ha="center", va="center", fontsize=FS_TICK - 1, color=INK)
        ax.add_patch(patches.Rectangle((x_mean, i), 1, 1, facecolor=WIN_CMAP(row_mean[i]),
                                       edgecolor=SURFACE, lw=2))
        ax.text(x_mean + 0.5, i + 0.5, fmt(row_mean[i]), ha="center", va="center",
                fontsize=FS_TICK - 1, color=INK)
        ax.text(-0.15, i + 0.5, MLAB[EVAL_ARMS[i]], ha="right", va="center", fontsize=FS_TICK, color=INK)
    for j in range(n):
        ax.text(j + 0.5, -0.15, MLAB[EVAL_ARMS[j]], ha="center", va="bottom", rotation=90,
                fontsize=FS_TICK - 1, color=INK)
    ax.text(x_mean + 0.5, -0.15, "row mean", ha="center", va="bottom", rotation=90,
            fontsize=FS_TICK - 1, color=INK)

    # the rotated column names live INSIDE the y range, so the axes box (equal
    # aspect, anchored to the top of its cell) starts at the top of the names
    # and the panel is flush with (d) and (f); the colorbar is an inset right
    # under the matrix instead of stealing space at the bottom of the cell
    renderer = fig.canvas.get_renderer()
    h_px = max(t.get_window_extent(renderer=renderer).height for t in ax.texts
               if t.get_rotation() == 90)
    w_in = ax.get_position().width * fig.get_figwidth()
    lab_h = (h_px / fig.dpi) * (x_mean + 1.10) / w_in       # data units (cells) above the matrix
    ax.set_xlim(-0.05, x_mean + 1.05)
    ax.set_ylim(n + 0.05, -(0.15 + lab_h + 0.05))
    ax.set_aspect("equal", anchor="N")
    ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap=WIN_CMAP, norm=mc.Normalize(0, 1))
    cax = ax.inset_axes([0.0, -0.055, 1.0, 0.032])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Win rate of row\nover column (.5 = tie)", fontsize=FS_TICK - 1, color=INK2, labelpad=4)
    cb.ax.tick_params(labelsize=FS_TICK - 2)
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_ticklabels(["0", ".5", "1"])
    cb.outline.set_visible(False)


# ── (c) PACS breadth, one row per hierarchy level ─────────────────────────
LEVEL_NAME = {0: "root", 1: "field", 2: "division", 3: "subdivision"}
LEVEL_ROWS = ["field", "division", "subdivision"]          # top to bottom
LEVEL_SHORT = {"field": "field", "division": "div.", "subdivision": "sub."}
# Short official PACS names where the eval's gt_label is long; every other
# node uses its gt_label from label_eval_metric1.json as is.
OFFICIAL_SHORT = {
    "8": "Interdisciplinary physics",          # not among the 28 scored nodes
    "03": "Quantum mech. & relativity",        # Quantum mechanics, field theories, and special relativity
    "05": "Statistical physics",               # ..., thermodynamics, and nonlinear dynamical systems
    "11": "Fields & particles",                # General theory of fields and particles
    "12": "Interaction models",                # Specific theories and interaction models; particle systematics
    "71": "Electronic structure",              # Electronic structure of bulk materials
    "75": "Magnetism",                         # Magnetic properties and materials
    "05.40": "Fluctuations & noise",           # Fluctuation phenomena, random processes, noise, and Brownian motion
    "05.45": "Nonlinear dynamics",             # Nonlinear dynamics and chaos
    "11.10": "Fields & particles", "11.15": "Fields & particles",
    "12.60": "Interaction models",
    "71.10": "Many-electron systems",          # Theories and models of many-electron systems
    "71.20": "Electronic structure",
    "75.10": "Magnetic ordering",              # General theory and models of magnetic ordering
    "75.30": "Magnetism",
}
# named first whatever the room: the nodes the text discusses
CALLOUT_FIRST = ["8", "3", "6", "47", "05", "03.65", "12.38"]


def _same_field(decoded, official):
    """'Condensed matter physics' names the same node as 'Condensed matter';
    'Quantum optics' is a name for 'Optics'."""
    norm = lambda t: t.lower().replace(" physics", "").strip()
    a, b = norm(decoded), norm(official)
    return a == b or a.endswith(" " + b) or b.endswith(" " + a)


def panel_pacs(ax, fig, radius_csv, labels_json, fuzzy_json):
    df = pd.read_csv(radius_csv)
    df["node"] = df["node"].astype(str)
    df = df[df["node"] != "root"].copy()
    df["depth"] = df["level"].map(LEVEL_NAME)
    df = df.rename(columns={"n": "n_papers", "centroid_norm": "radius"})
    labels = json.load(open(labels_json))
    official = {n["code"]: n["gt_label"] for n in json.load(open(fuzzy_json))["per_node"]}
    official.update(OFFICIAL_SHORT)
    nmin, nmax = float(df.n_papers.min()), float(df.n_papers.max())

    def msize(n):
        return 40 + 520 * ((n ** 0.5 - nmin ** 0.5) / max(nmax ** 0.5 - nmin ** 0.5, 1e-9))

    y_of = {lv: len(LEVEL_ROWS) - 1 - i for i, lv in enumerate(LEVEL_ROWS)}
    xmin, xmax = df.radius.min() - 0.03, df.radius.max() + 0.17   # the extra x room is for text only
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.28, len(LEVEL_ROWS) - 1 + 1.22)                  # the top row stacks five callouts
    ux, uy = _units_per_inch(ax, fig)
    fs = FS_TEXT - 1
    line_h = 1.25 * fs / 72 * uy
    tier_h = line_h + 0.03 * uy
    pad = 0.05 * ux
    named = []

    for lv in LEVEL_ROWS:
        y = y_of[lv]
        sub = df[df.depth == lv]
        ax.plot([xmin, 3.0], [y, y], color=GREY_FILL, lw=1.8, zorder=1)
        ax.text(xmin - 0.01, y, LEVEL_SHORT[lv], ha="right", va="center", fontsize=FS_TICK, color=INK)
        # candidates: every node of the row with a decoded label and an official name
        items = []
        x_lo, x_hi = sub.radius.min(), sub.radius.max()
        for _, r in sub.iterrows():
            if r.node not in labels or r.node not in official:
                continue
            dec = abbrev(labels[r.node])
            off = None if _same_field(labels[r.node], official[r.node]) else f"({abbrev(official[r.node])})"
            w_dec = _text_w(ax, dec, fs) * ux
            variants = [(w_dec, 1)]
            if off:
                w_off = _text_w(ax, off, fs) * ux
                variants = [(w_dec + pad + w_off, 1), (max(w_dec, w_off), 2)]
            rank = 0 if r.node in CALLOUT_FIRST else 1 if r.radius in (x_lo, x_hi) else 2
            items.append({"x": float(r.radius), "node": r.node, "dec": dec, "off": off,
                          "variants": variants, "prio": (rank, -float(r.n_papers))})
        room = (1.22 if lv == LEVEL_ROWS[0] else 1.0) - 0.14 * uy - 0.10 * uy   # data units above this line
        max_tiers = int(room // tier_h)
        placed = pack_callouts(items, xmin, xmax, ux, max_tiers, gap_in=0.10, inset_in=0.03)
        hi = {p["node"] for p in placed}
        named += sorted(hi)
        for _, r in sub.iterrows():
            is_hi = r.node in hi
            ax.scatter([r.radius], [y], s=msize(float(r.n_papers)), color=ORANGE if is_hi else GREY_MARK2,
                       edgecolors=INK if is_hi else SURFACE, linewidths=1.2 if is_hi else 0.6,
                       zorder=4 if is_hi else 3)
        for p in placed:
            y_base = y + 0.14 * uy + p["tier"] * tier_h
            draw_leader(ax, p["x"], y, y_base - 0.015 * uy, uy)
            if p["h"] == 1:
                t = ax.text(p["s"], y_base, p["dec"], ha="left", va="bottom", fontsize=fs, color=ORANGE, zorder=5)
                if p["off"]:
                    x_end = ax.transData.inverted().transform(
                        (t.get_window_extent(renderer=fig.canvas.get_renderer()).x1, 0))[0]
                    ax.text(x_end + pad, y_base, p["off"], ha="left", va="bottom", fontsize=fs, color=INK2, zorder=5)
            else:                                                   # decoded above its official name
                ax.text(p["s"], y_base + line_h, p["dec"], ha="left", va="bottom", fontsize=fs, color=ORANGE, zorder=5)
                ax.text(p["s"], y_base, p["off"], ha="left", va="bottom", fontsize=fs, color=INK2, zorder=5)

    ax.set_xticks([2.7, 2.8, 2.9, 3.0])
    ax.set_xticklabels(["2.7", "2.8", "2.9", "3.0"], fontsize=FS_TICK)
    ax.spines["bottom"].set_bounds(xmin, 3.0)
    ax.set_yticks([])
    ax.set_xlabel(r"vector length  $\|\bar{v}\|_2$", fontsize=FS_LABEL)
    sns.despine(ax=ax, left=True)
    return named


# ── (d) abstraction walk: one horizontal line per seed document ───────────
WALK_ORDER = ["physics concept", "biochemistry", "fruit", "patent (LED)"]
WALK_SHORT = {"physics concept": "physics", "biochemistry": "biochem.", "fruit": "fruit",
              "patent (LED)": "patent"}
WALK_GREY = np.array(mc.to_rgb(GREY_MARK2))
ROW_PITCH = 1.25      # vertical distance between lines (data units)
WALK_TIERS = 2        # label tiers per line; a run whose label finds no room is left unlabelled
# a run of equal labels is named once, on its outermost step (the largest alpha)


def _fade(color, t):
    rgb = np.array(mc.to_rgb(color))
    return tuple((1 - float(t)) * rgb + float(t) * WALK_GREY)


def _runs(steps):
    """Consecutive steps (outermost first) with the same label -> [(label, [alphas])]."""
    runs = []
    for s in steps:
        if runs and s["label"].lower() == runs[-1][0].lower():
            runs[-1][1].append(s["alpha"])
        else:
            runs.append((s["label"], [s["alpha"]]))
    return runs


def panel_walk(ax, fig, walk_json):
    walk = json.load(open(walk_json))
    alphas = [s["alpha"] for s in walk[WALK_ORDER[0]]["steps"]]     # the same grid for every seed
    amax = max(alphas)
    fs = FS_TEXT - 1
    n_rows = len(WALK_ORDER)
    y_of = {e: (n_rows - 1 - i) * ROW_PITCH for i, e in enumerate(WALK_ORDER)}
    xmin, xmax = 0.0, amax * 1.08
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.8, y_of[WALK_ORDER[0]] + 1.2)
    ux, uy = _units_per_inch(ax, fig)
    line_h = 1.25 * fs / 72 * uy
    tier_h = line_h + 0.06 * uy
    prior = walk[WALK_ORDER[0]]["steps"][-1]["label"]      # what every seed ends at
    band_h = 0.30 * ROW_PITCH
    dropped = []

    for e in WALK_ORDER:
        y = y_of[e]
        steps = walk[e]["steps"]
        runs = _runs(steps)
        # the innermost run is the base-model prior: a band on this line up to
        # the midpoint between its outermost step and the next label
        assert runs[-1][0].lower() == prior.lower(), (e, runs[-1][0], prior)
        x_band = 0.5 * (max(runs[-1][1]) + min(runs[-2][1]))
        ax.add_patch(patches.Rectangle((0, y - band_h), x_band, 2 * band_h, facecolor=GREY_LIGHT,
                                       edgecolor="none", zorder=0))
        ax.plot([alphas[-1], amax], [y, y], color=GREY_FILL, lw=1.8, zorder=1)
        for st in steps:
            a = st["alpha"]
            ax.scatter([a], [y], s=80, color=_fade(ORANGE, 1 - a / amax), edgecolors=SURFACE,
                       linewidths=0.9, zorder=3)
        ax.text(-0.02 * xmax, y, WALK_SHORT[e], ha="right", va="center", fontsize=FS_TICK, color=INK)
        items = []
        for k, (lab, xs) in enumerate(runs[:-1]):             # outermost first; the prior run has no label
            text = abbrev(lab)
            items.append({"x": max(xs), "text": text,
                          "variants": [(_text_w(ax, text, fs) * ux, 1)], "prio": k})
        placed = pack_callouts(items, xmin, xmax, ux, WALK_TIERS, gap_in=0.18)
        got = {p["text"] for p in placed}
        dropped += [(e, it["text"]) for it in items if it["text"] not in got]
        for p in placed:
            y_base = y + 0.10 * uy + p["tier"] * tier_h
            draw_leader(ax, p["x"], y, y_base - 0.015 * uy, uy)
            ax.text(p["s"], y_base, p["text"], ha="left", va="bottom", fontsize=fs, color=INK, zorder=5)

    y0 = y_of[WALK_ORDER[-1]]
    ax.text(0.007 * xmax, y0 - band_h - 0.03 * uy, f"base-model prior ({prior.replace(' and ', ' & ')})",
            ha="left", va="top", fontsize=fs, color=INK2)
    # round ticks do not sit under the dots, so draw the tick marks (the seaborn
    # "white" style turns them off: xtick.bottom = False) for the reader to locate them
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", ".25", ".5", ".75", "1"], fontsize=FS_TICK)
    ax.tick_params(axis="x", bottom=True, length=7, width=1.0, color=INK2)
    ax.set_yticks([])
    ax.set_xlabel(r"scale $\alpha$  ($\|v\|_2 = \alpha\,\|v_{\mathrm{doc}}\|_2$)", fontsize=FS_LABEL)
    sns.despine(ax=ax, left=True)
    return dropped


# ── (e) mixing fidelity, (f) copy rate ────────────────────────────────────
def curves(ax, cache_json, t2l_json, kind, stratum, ylabel, ylim, ideal=False, baseline=None,
           legend=False):
    out = json.load(open(cache_json))
    sets = [s for s, v in out.items() if v["stratum"] == stratum]
    if not sets:
        raise ValueError(f"no pair sets with stratum {stratum!r} in {cache_json}")
    keys = sorted({k for s in sets for m in out[s]["pts"].values() for k in m}, key=float)
    ts = [float(k) for k in keys]
    if ideal:
        ax.plot([0, 1], [0, 1], "--", color=GREY_MARK2, lw=1.4, zorder=1)
        ax.text(0.22, 0.25, "ideal", rotation=45, ha="center", va="bottom", fontsize=FS_TICK - 2,
                color=INK2, rotation_mode="anchor")
    if baseline is not None:
        ax.axhline(baseline, ls="--", color=GREY_MARK2, lw=1.4, zorder=1)
        ax.text(0.02, baseline - 0.015, f"random (~{int(baseline * 100)}%)", fontsize=FS_TICK - 3,
                color=INK2, ha="left", va="top")
    handles = {}
    for ch in ["doc2lora", "icae", "incontext"]:
        m, lo, hi = [], [], []
        for k in keys:
            vals = [out[s]["pts"][ch][k] for s in sets if k in out[s]["pts"].get(ch, {})]
            a, b, c = boot_ci(vals)
            m.append(a); lo.append(b); hi.append(c)
        z = 6 if ch == "doc2lora" else 3
        ax.fill_between(ts, lo, hi, color=LINE_COLOR[ch], alpha=0.18, lw=0, zorder=z - 1)
        handles[ch] = ax.plot(ts, m, "-o", color=LINE_COLOR[ch], ms=5.5,
                              lw=3.2 if ch == "doc2lora" else 2.4, mec=SURFACE, mew=0.8, zorder=z)[0]
        handles[ch].end = m[-1]
    # T2L: decoded on the same alpha grid as the other arms (t2l_fusion.py)
    t2l = json.load(open(t2l_json))
    tsets = [s for s, v in t2l.items() if not s.startswith("_") and v["stratum"] == stratum]
    tkeys = sorted({k for s in tsets for k in t2l[s]["pts"][kind]}, key=float)
    m, lo, hi = [], [], []
    for k in tkeys:
        a, b, c = boot_ci([t2l[s]["pts"][kind][k] for s in tsets if k in t2l[s]["pts"][kind]])
        m.append(a); lo.append(b); hi.append(c)
    tx = [float(k) for k in tkeys]
    ax.fill_between(tx, lo, hi, color=LINE_COLOR["t2l"], alpha=0.25, lw=0, zorder=2)
    handles["t2l"] = ax.plot(tx, m, ":s", color=LINE_COLOR["t2l"], ms=5.5, lw=2.4, mec=SURFACE,
                             mew=0.8, zorder=3)[0]
    handles["t2l"].end = m[-1]

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(*ylim)
    if legend:                                             # top-left, arms ordered as they end
        order = sorted(handles, key=lambda c: -handles[c].end)
        ax.legend([handles[c] for c in order], [MLAB[c] for c in order], loc="upper left",
                  frameon=False, fontsize=FS_LEGEND, handlelength=1.8, borderaxespad=0.1,
                  labelspacing=0.25, handletextpad=0.5)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", ".5", "1"], fontsize=FS_TICK)
    ax.set_xlabel("Ideal mixing", fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    sns.despine(ax=ax)
    return len(sets), len(tsets)


# ── assembly ──────────────────────────────────────────────────────────────
def build(fuzzy_json, judge_json, radius_csv, labels_json, walk_json, metrics_json, copyrate_json,
          t2l_json, out_pdf, out_png=None, stratum="L1"):
    set_style()
    m1 = json.load(open(fuzzy_json))
    arm_keys = {a: a for a in EVAL_ARMS}
    arm_keys["t2l"] = _t2l_space(m1)

    FIG_W, LEFT, RIGHT = 16.4, 0.05, 0.99
    fig = plt.figure(figsize=(FIG_W, 12))
    # three panel columns with explicit spacer columns between them (wspace = 0),
    # so each gap is set on its own; the widths are inches of the span. The outer
    # columns are equal; the gap before (c)/(d) also holds the walk's row names
    # ("biochem."), so it is wider than the gap before (e)/(f); the middle column
    # takes the rest (5.78 in: (c) keeps its 14 callouts down to ~5.7 in).
    W_SIDE, GAP_12, GAP_23 = 3.45, 1.56, 1.18
    W_MID = (RIGHT - LEFT) * FIG_W - 2 * W_SIDE - GAP_12 - GAP_23
    gs = GridSpec(2, 5, figure=fig, width_ratios=[W_SIDE, GAP_12, W_MID, GAP_23, W_SIDE],
                  height_ratios=[1.10, 0.90],
                  left=LEFT, right=RIGHT, top=0.97, bottom=0.055, wspace=0.0, hspace=0.24)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 2])
    ax_e = fig.add_subplot(gs[0, 4])
    ax_f = fig.add_subplot(gs[1, 4])

    panel_fuzzy(ax_a, fuzzy_json, arm_keys)
    panel_judge(ax_b, fig, judge_json, arm_keys)
    named = panel_pacs(ax_c, fig, radius_csv, labels_json, fuzzy_json)
    dropped = panel_walk(ax_d, fig, walk_json)
    n_e, n_t = curves(ax_e, metrics_json, t2l_json, "mix", stratum, "Actual mixing", (-0.03, 1.03),
                      ideal=True, legend=True)
    n_f, _ = curves(ax_f, copyrate_json, t2l_json, "copy", stratum, "Copy rate", (0.25, 0.85), baseline=0.30)

    # panel letters at the top-left corner of each GridSpec cell, so the three
    # letters of a row share one baseline whatever the axes inside do
    for (r, c), letter in zip([(0, 0), (1, 0), (0, 2), (1, 2), (0, 4), (1, 4)], "abcdef"):
        bb = gs[r, c].get_position(fig)
        fig.text(bb.x0 - 0.035, bb.y1 + 0.004, f"({letter})", fontsize=FS_PANEL, fontweight="bold",
                 ha="left", va="bottom")

    os.makedirs(os.path.dirname(os.path.abspath(out_pdf)), exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05, transparent=True)
    if out_png:
        fig.savefig(out_png, dpi=110, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_pdf}  (T2L = {arm_keys['t2l']}; panels e/f: {stratum}, n={n_e} pairs, T2L n={n_t})")
    print(f"  (c) named {len(named)} nodes: {' '.join(named)}")
    print(f"  (d) unlabelled runs: {dropped or 'none'}")


if "snakemake" in sys.modules:
    build(fuzzy_json=snakemake.input["fuzzy_json"],
          judge_json=snakemake.params["judge_json"],   # a param on purpose; see the rule
          radius_csv=snakemake.input["radius_csv"],
          labels_json=snakemake.input["labels_json"],
          walk_json=snakemake.input["walk_json"],
          metrics_json=snakemake.input["metrics_json"],
          copyrate_json=snakemake.input["copyrate_json"],
          t2l_json=snakemake.input["t2l_json"],
          out_pdf=snakemake.output["pdf"],
          stratum=snakemake.params["stratum"])
elif __name__ == "__main__":
    BT = "data/labels"
    SKG = "data/pair_axis"
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuzzy-json", default=f"{BT}/label_eval_metric1.json")
    ap.add_argument("--judge-json", default=f"{BT}/label_eval_metric4.json")
    ap.add_argument("--radius-csv", default=f"{BT}/length_vs_breadth.csv")
    ap.add_argument("--labels-json", default=f"{BT}/qwen_fullrank_field23.json")
    ap.add_argument("--walk-json", default=f"{BT}/abstraction_walk.json")
    ap.add_argument("--metrics-json", default=f"{SKG}/results/pair_axis_metrics_pairaxis.json")
    ap.add_argument("--copyrate-json", default=f"{SKG}/results/pair_axis_copyrate_pairaxis_L1.json")
    ap.add_argument("--t2l-json", default=f"{SKG}/results/pair_axis_t2l_e1_pairaxis.json")
    ap.add_argument("--stratum", default="L1")
    ap.add_argument("--out", default="figs/pacs-clustering.pdf")
    ap.add_argument("--png", default=None, help="optional raster preview")
    a = ap.parse_args()
    build(a.fuzzy_json, a.judge_json, a.radius_csv, a.labels_json, a.walk_json, a.metrics_json,
          a.copyrate_json, a.t2l_json, a.out, a.png, a.stratum)
