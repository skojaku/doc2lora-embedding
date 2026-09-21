"""Two-corner mixing colorband: measured blend (blue<->orange, colorblind) along
the A-B interpolation axis, one discrete rectangle per decoded cell (13 points,
GRID_DEN=12), colored by the SBERT-projected actual mixing fraction (same
projection as pair_axis_metrics.py). Decoded keywords are shown below the band
at the 25%/50%/75% cells. The two SOURCE corners' real identity (paper title +
DOI, not decoded) flanks the band at the far left/right, colored red (A) /
orange (B) to match the band's right (orange) end.

  python pair_axis_colorband.py pairL1_00                       # all 3 methods, stacked
  python pair_axis_colorband.py pairL1_00 --method doc2lora      # single method
"""
import os, sys, json, re, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simplex_common as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
CB = sns.color_palette("colorblind")
BLUE, ORANGE, RED = np.array(CB[0]), np.array(CB[1]), np.array([0.85, 0.10, 0.10])
LAB = {"doc2lora": "Doc2LoRA", "icae": "ICAE", "incontext": "In-context"}
PAD = 0.42    # horizontal margin reserved for the pole (KeyBERT) keywords + method label
FIG_W = 17.0
NOMINAL_ROW_H = 4.6   # design row height; band cell aspect is pinned to this regardless of --row-scale
BAND_Y0, BAND_H = 0.535, 0.23   # half the original height, centered on the same midline

# blue -> orange straight RGB lerp passes through a muddy desaturated gray/olive at the
# midpoint (they're near-complementary), which hides the gradation. Route through white
# instead (a standard diverging-colormap trick) so the transition stays visibly saturated.
from matplotlib.colors import LinearSegmentedColormap
_MID = (0.84, 0.84, 0.82)   # off-white, not pure white -- a pure-white cell vanishes against the page
_CMAP = LinearSegmentedColormap.from_list("blue_white_orange", [BLUE, _MID, ORANGE])


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


POLE_GAMMA = 2.2   # >1: stretch color contrast near the poles (0/1), compress it near the center


def blend(t):
    t = float(np.clip(t, 0, 1))
    u = 2 * (t - 0.5)                                  # [-1, 1]
    u = np.sign(u) * abs(u) ** POLE_GAMMA              # stretch near |u|=1, compress near 0
    return _CMAP(0.5 + u / 2)


_COPY_CMAP = cm.get_cmap("cividis")


def toks(t):
    return re.findall(r"[a-z][a-z-]{2,}", (t or "").lower())


COPY_GAMMA = 0.75   # <1: extra contrast on top of the min-max stretch


def copy_color(rate_rescaled):
    x = float(np.clip(rate_rescaled, 0, 1)) ** COPY_GAMMA
    return _COPY_CMAP(x)


def cap(w):
    return w[0].upper() + w[1:] if w else w


KW_WRAP_WIDTH = 20   # chars; fixed column width so adjacent keyword blocks never collide


def wrap_kw(items, width=KW_WRAP_WIDTH):
    import textwrap
    joined = ", ".join(cap(w) for w in items)
    return "\n".join(textwrap.wrap(joined, width, break_long_words=False))


def kw_list_from_decoded(kw_str, top_n=3):
    return [w.strip() for w in kw_str.split(",") if w.strip()][:top_n]


def lookup_paper_ids(set_name):
    for mf in ["pair_manifest.json", "mild_manifest.json"]:
        p = os.path.join(HERE, mf)
        if not os.path.exists(p):
            continue
        man = json.load(open(p))
        rec = next((m for m in man if m["set"] == set_name), None)
        if rec:
            return rec["paper_ids"]
    return None


def lookup_doi_title(set_name):
    ids = lookup_paper_ids(set_name)
    if not ids:
        return None, None
    import pandas as pd
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    df = pd.read_parquet(os.path.join(root, "data/aps/paper_text.parquet"),
                         columns=["aps_paper_id", "doi", "title"])
    df = df.set_index("aps_paper_id")
    out = []
    for pid in ids[:2]:
        if pid in df.index:
            row = df.loc[pid]
            out.append((row["title"], row["doi"]))
        else:
            out.append((None, None))
    return out


def method_data(set_name, method, tag, emb, eA, axis, denom, srctok=None, metric="mix"):
    R = sc.GRID_DEN
    suf = "_icae" if method == "icae" else ""
    fpath = os.path.join(sc.RESULTS, f"absfollow_{set_name}{tag}{suf}.json")
    cells = {tuple(c["bary"]): c for c in json.load(open(fpath))["cells"]}
    values, kw_by_k = [], {}
    for k in range(R + 1):
        c = cells[(R - k, k, 0)]   # k=0 -> pure A (weight_B=0), k=R -> pure B (weight_B=1)
        rec = c.get(method, {})
        ab, kw = rec.get("abs", ""), rec.get("kw", "")
        if metric == "copy":
            at = toks(ab)
            rate = sum(w in srctok for w in at) / len(at) if at else 0.0
            values.append(rate)
        else:
            d = emb(ab)
            t = float((d - eA) @ axis / denom)
            values.append(t)
        kw_by_k[k] = kw
    return values, kw_by_k


def plot_row(ax, R, cols, kw_by_k, label, meta=None, show_axis=True, metric="mix",
             show_keywords=False, show_endpoints=False, copy_range=(0.0, 1.0), summary=None):
    n = R + 1
    pad = PAD   # fixed regardless of keywords/endpoints, so cell aspect ratio matches across figures
    for k in range(n):
        ax.add_patch(patches.Rectangle((k / n, BAND_Y0), 1 / n, BAND_H,
                                       facecolor=cols[k], edgecolor="0.55", lw=0.7))
    ax.set_xlim(-pad, 1 + pad); ax.set_ylim(0, 1)
    # aspect chosen so each cell (width 1/n, height BAND_H in data units) renders SQUARE in
    # physical inches, regardless of figure width/height or --row-scale: mpl's default
    # aspect='auto' would otherwise re-stretch the fixed data range to fill whatever box it's given.
    ax.set_aspect(1 / (n * BAND_H), adjustable="box")
    ax.axis("off")

    quarter_k = {0.25: round(0.25 * R), 0.5: round(0.5 * R), 0.75: round(0.75 * R)}
    TEXT_TOP = BAND_Y0 - 0.16
    LINE_MARGIN = 0.025   # gap between the guide line's end and the keyword text's top edge
    band_top = BAND_Y0 + BAND_H

    # method label sits directly above its own band
    label_y = band_top + 0.045
    ax.text(0.5, label_y, label, transform=ax.transData, fontsize=26,
            fontweight="bold", ha="center", va="bottom")

    if show_axis:
        axis_y = label_y + 0.20   # cleared to sit above the (larger) method label
        ax.plot([0, 1], [axis_y, axis_y], color="0.3", lw=1, zorder=3)
        for frac in [0, 0.25, 0.5, 0.75, 1.0]:
            ax.plot([frac, frac], [axis_y - 0.015, axis_y + 0.015], color="0.3", lw=1, zorder=3)
            ax.text(frac, axis_y + 0.025, f"{int(frac * 100)}%", ha="center", va="bottom",
                    fontsize=20, color="0.3")
        ax.text(0.5, axis_y + 0.16, "ideal mixing weight  (A → B)", ha="center", va="bottom",
                fontsize=22, color="0.15")

    if show_axis and metric == "copy":
        # cividis legend strip for what cell color means (independent of x-position).
        # anchored relative to axis_y (data coords) instead of a stale fixed axes-fraction,
        # so it sits just above the axis title regardless of row_h.
        cbar_y0 = axis_y + 0.52
        cbar_ax = ax.inset_axes([0.20, cbar_y0, 0.60, 0.11], transform=ax.transData)
        grad = np.linspace(0, 1, 256).reshape(1, -1)
        cbar_ax.imshow(grad, aspect="auto", cmap=_COPY_CMAP, extent=[0, 1, 0, 1])
        cbar_ax.set_xticks([0, 1])
        cbar_ax.set_xticklabels([f"{int(copy_range[0] * 100)}% copy", f"{int(copy_range[1] * 100)}% copy"],
                                fontsize=20)
        cbar_ax.set_yticks([])
        for spine in cbar_ax.spines.values():
            spine.set_visible(False)

    if show_keywords:
        SHIFT = 4.4 / n   # 25% keywords move left, 75% move right -- more breathing room from center
        text_x = {0.25: 0.25 - SHIFT, 0.5: 0.5, 0.75: 0.75 + SHIFT}
        for frac in [0.25, 0.5, 0.75]:
            k = quarter_k[frac]
            cx = (k + 0.5) / n
            if metric != "copy":
                ax.plot([cx, text_x[frac]], [BAND_Y0 + BAND_H / 2, TEXT_TOP + LINE_MARGIN],
                        color="0.6", lw=1, zorder=2)
            ax.plot(cx, BAND_Y0 + BAND_H / 2, "o", ms=9, mfc="white", mec="black", mew=1.2, zorder=5)

        def entry_text(frac):
            if summary and str(frac) in summary:
                import textwrap
                return "\n".join(textwrap.wrap(summary[str(frac)], KW_WRAP_WIDTH, break_long_words=False))
            return wrap_kw(kw_list_from_decoded(kw_by_k[quarter_k[frac]]))

        mid_entries = [
            (text_x[0.25], entry_text(0.25), "0.15", "center"),
            (text_x[0.5], entry_text(0.5), "0.15", "center"),
            (text_x[0.75], entry_text(0.75), "0.15", "center"),
        ]
        for x, text, color, ha in mid_entries:
            ax.text(x, TEXT_TOP, text, transform=ax.transData, fontsize=20, color=color,
                    ha=ha, va="top", linespacing=1.5)

    # endpoint annotation = source paper's title (larger) + DOI (smaller), not decoded --
    # the true identity of each corner. Same for every method, so only show once.
    if show_endpoints and meta is not None:
        import textwrap
        (titleA, doiA), (titleB, doiB) = meta
        yc = BAND_Y0 + BAND_H / 2
        if titleA:
            ax.text(-0.06, yc + 0.02, "\n".join(textwrap.wrap(titleA, 24)), transform=ax.transData,
                    fontsize=18, color=BLUE, ha="right", va="bottom", linespacing=1.3)
            if doiA:
                ax.text(-0.06, yc - 0.02, doiA, transform=ax.transData,
                        fontsize=16, color=BLUE, ha="right", va="top", style="italic")
        if titleB:
            ax.text(1.06, yc + 0.02, "\n".join(textwrap.wrap(titleB, 24)), transform=ax.transData,
                    fontsize=18, color=ORANGE, ha="left", va="bottom", linespacing=1.3)
            if doiB:
                ax.text(1.06, yc - 0.02, doiB, transform=ax.transData,
                        fontsize=16, color=ORANGE, ha="left", va="top", style="italic")


def main(set_name, methods, tag, metric="mix", show_keywords=False, show_endpoints=False, row_scale=1.0):
    R = sc.GRID_DEN
    spec = sc.load_corners(sc.corners_path(set_name))
    A, B = spec["leads"]["A"], spec["leads"]["B"]

    from sentence_transformers import SentenceTransformer
    sb = SentenceTransformer("all-mpnet-base-v2", device="cuda")

    def emb(t):
        return unit(np.atleast_2d(sb.encode(t, normalize_embeddings=True, show_progress_bar=False)).ravel())

    eA, eB = emb(A), emb(B)
    axis = eB - eA
    denom = float(axis @ axis) + 1e-12
    srctok = set(toks(A)) | set(toks(B))

    meta = lookup_doi_title(set_name)

    # optional hand-written one-sentence summaries (brevity over grammar), keyed by
    # set_name -> method -> "0.25"/"0.5"/"0.75"; falls back to raw decoded keywords if absent.
    summaries_path = os.path.join(HERE, "keyword_summaries.json")
    all_summaries = json.load(open(summaries_path)) if os.path.exists(summaries_path) else {}
    set_summaries = all_summaries.get(set_name, {})

    # gather raw values first (t for mix, copy-rate for copy). Copy rates cluster in a narrow
    # band so a fixed 0-1 cividis scale barely shows any difference -- rescale a fixed 30%-70%
    # window (not the data's own min/max) to the full colormap for consistent, comparable colors
    # across different pairs/figures.
    per_method = {}
    for method in methods:
        values, kw_by_k = method_data(set_name, method, tag, emb, eA, axis, denom,
                                      srctok=srctok, metric=metric)
        per_method[method] = (values, kw_by_k)

    copy_lo, copy_hi = 0.40, 0.80   # 0.40 ~ above the ~30% random-chance token-overlap floor

    row_h = NOMINAL_ROW_H * row_scale
    fig, axes = plt.subplots(len(methods), 1, figsize=(FIG_W, row_h * len(methods)))
    if len(methods) == 1:
        axes = [axes]
    for i, (ax, method) in enumerate(zip(axes, methods)):
        values, kw_by_k = per_method[method]
        if metric == "copy":
            cols = [copy_color((v - copy_lo) / (copy_hi - copy_lo + 1e-9)) for v in values]
        else:
            cols = [blend(v) for v in values]
        plot_row(ax, R, cols, kw_by_k, LAB[method], meta=meta if i == 0 else None,
                show_axis=(i == 0), metric=metric,
                show_keywords=show_keywords, show_endpoints=show_endpoints,
                copy_range=(copy_lo, copy_hi), summary=set_summaries.get(method))

    fig.tight_layout()
    if show_keywords:
        # keyword blocks wrap to several lines (narrow fixed width to avoid horizontal
        # collision) -- tight_layout doesn't leave enough row/top/bottom margin for that
        # (ax.set_aspect("box") re-anchors each axes after tight_layout sized the slots),
        # so widen the gaps and margins explicitly.
        fig.subplots_adjust(hspace=1.3, top=0.92, bottom=0.16)
    tag_name = "_".join(methods) if len(methods) > 1 else methods[0]
    msuf = "_copyrate" if metric == "copy" else ""
    out = os.path.join(HERE, "figs", f"colorband_{set_name}_{tag_name}{msuf}.pdf")
    # NOT bbox_inches="tight" -- it crops to each figure's own content extent (colorbar vs
    # endpoint text overflow differently), giving mix/copy-rate PDFs different canvas sizes
    # even though the cells are geometrically identical. Fixed canvas keeps them pixel-consistent.
    fig.savefig(out, dpi=300, transparent=True)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("set_name")
    ap.add_argument("--method", default=None, choices=["doc2lora", "incontext", "icae"],
                    help="single method; omit to plot all 3 (doc2lora, icae, incontext) stacked")
    ap.add_argument("--tag", default="_pairaxis")
    ap.add_argument("--metric", default="mix", choices=["mix", "copy"],
                    help="mix = SBERT blend position (default); copy = verbatim copy rate (cividis)")
    ap.add_argument("--keywords", action="store_true", help="show decoded keywords at 25/50/75%%")
    ap.add_argument("--endpoints", action="store_true", help="show source paper title+DOI at the poles")
    ap.add_argument("--row-scale", type=float, default=1.0,
                    help="multiply row height (keeps width fixed at 17in), e.g. 2.0 to double figure height")
    args = ap.parse_args()
    methods = [args.method] if args.method else ["doc2lora", "icae", "incontext"]
    main(args.set_name, methods, args.tag, metric=args.metric,
        show_keywords=args.keywords, show_endpoints=args.endpoints, row_scale=args.row_scale)
