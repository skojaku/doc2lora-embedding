"""Per-cell RGB + copy-rate from the ABSTRACTS decoded by decode_absfollow.py.

  RGB       = soft responsibility of the abstract embedding to the 3 corner
              centroids (mean SBERT of the corner keyphrases), gamma-sharpened.
  COPY-RATE = fraction of abstract word-tokens appearing verbatim in the 3
              source lead abstracts (extractive / verbatim overlap).

The follow-up keywords are NOT used here (they are for figure clarity only).

  CUDA_VISIBLE_DEVICES=0 KWTAG=_kwsrc python absf_metrics.py strat00 strat80
  KWTAG=_kwsrc python absf_metrics.py --replot strat00 strat80          # no GPU
"""
import os, sys, json, re, ast
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simplex_common as sc
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflow", "plot"))
import _simplex as S
from _style import setup_style
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = ["doc2lora", "incontext", "icae"]
LAB = {"ideal": "Ideal (weights)", "doc2lora": "Doc2LoRA", "incontext": "In-context", "icae": "ICAE"}
SUF = os.environ.get("KWTAG", "_kwsrc")
TAU = float(os.environ.get("TAU", "0.30"))
GAMMA = float(os.environ.get("GAMMA", "1.5"))
CENTER = os.environ.get("CENTER", "0") == "1"   # subtract 3-corner centroid before cosine, re-unit
CELL_SIZE, CELL_EDGE, CELL_EDGE_LW, WHITEN = 110, "0.5", 0.28, 0.04


def toks(t):
    return re.findall(r"[a-z][a-z-]{2,}", (t or "").lower())


def as_words(v):
    if isinstance(v, list):
        return v
    try:
        return ast.literal_eval(v)
    except Exception:
        return [w for w in re.split(r"[,\n;]+", v) if w.strip()]


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def sharpen(w):
    w = np.clip(np.asarray(w, float), 0, None) ** GAMMA
    s = w.sum()
    return w / s if s > 0 else w


def resp(cos):
    e = np.exp((cos - cos.max()) / TAU)   # straight softmax of cosines, no mean-subtraction
    return e / e.sum()


IDW_EPS = 1e-6
IDW_P = float(os.environ.get("IDW_P", "2.0"))   # higher p -> sharper (more argmax-like), lower -> softer


DIST_KIND = os.environ.get("DIST_KIND", "cosine")   # "cosine" (1-cos, chordal) or "angular" (arccos/pi, geodesic)


def angdist(cos):
    """raw distance per corner, in [0,2] for cosine / [0,1] for angular, no normalization.
    DIST_KIND=cosine (default): d = 1 - cos  (chordal)
    DIST_KIND=angular:          d = arccos(cos) / pi  (geodesic on unit sphere)"""
    c = np.clip(np.asarray(cos, float), -1.0, 1.0)
    if DIST_KIND == "angular":
        return np.arccos(c) / np.pi
    return 1.0 - c


def idw_from_d(d):
    inv = 1.0 / (np.clip(np.asarray(d, float), 0, None) + IDW_EPS) ** IDW_P
    return inv / inv.sum()


def idw(cos):
    """inverse-distance weight: w_j -> 1 exactly as decode -> corner_j (d_j -> 0).
    unlike softmax, self-match isn't bounded away from 1 by the corner-pair similarity floor.
    d = angular distance (arccos(cos)/pi, in [0,1]) -- true geodesic on the unit sphere,
    not the chordal 1-cos approximation."""
    return idw_from_d(angdist(cos))


def compute(sets):
    from sentence_transformers import SentenceTransformer
    sb = SentenceTransformer("all-mpnet-base-v2", device="cuda")

    def emb(t):
        return unit(sb.encode(t, normalize_embeddings=True, show_progress_bar=False))

    out = {}
    for s in sets:
        spec = sc.load_corners(sc.corners_path(s))
        keys = spec["keys"]
        cent = np.stack([unit(np.atleast_2d(emb(spec["leads"][k])).ravel()) for k in keys])  # corner = original abstract
        ctr = cent.mean(0) if CENTER else 0.0
        cent_c = unit(cent - ctr) if CENTER else cent
        srctok = set()
        for k in keys:
            srctok |= set(toks(spec["leads"][k]))
        base = json.load(open(os.path.join(sc.RESULTS, f"absfollow_{s}{SUF}.json")))
        icp = os.path.join(sc.RESULTS, f"absfollow_{s}{SUF}_icae.json")
        icae = {tuple(c["bary"]): c["icae"] for c in json.load(open(icp))["cells"]} if os.path.exists(icp) else {}
        cellw, cellc, copy = {}, {}, {}
        for c in base["cells"]:
            b = tuple(c["bary"])
            src = {"doc2lora": c.get("doc2lora", {}), "incontext": c.get("incontext", {}),
                   "icae": icae.get(b, {})}
            for ch in METHODS:
                ab = (src[ch] or {}).get("abs", "")
                key = f"{ch}|{'-'.join(map(str,b))}"
                at = toks(ab)
                if len(at) < 8:
                    cellw[key] = None; cellc[key] = None; copy[key] = None; continue
                e = np.atleast_2d(emb(ab)).ravel()
                e_c = unit(e - ctr) if CENTER else e
                cos = (e_c @ cent_c.T).ravel()   # abstract vs 3 source abstracts
                cellw[key] = resp(cos).tolist()
                cellc[key] = cos.tolist()   # raw cosine, for re-weighting (idw etc.) w/o re-embedding
                copy[key] = sum(t in srctok for t in at) / len(at)
        out[s] = {"keys": keys, "names": [str(spec["names"][k])[:11] for k in keys],
                  "cellw": cellw, "cellc": cellc, "copy": copy}
    json.dump(out, open(os.path.join(sc.RESULTS, f"absf_metrics{SUF}.json"), "w"))
    return out


def frame(ax):
    for v0, v1 in [(S.VA, S.VB), (S.VB, S.VC), (S.VC, S.VA)]:
        ax.plot([v0[0], v1[0]], [v0[1], v1[1]], "-", color="0.22", lw=0.7, zorder=4)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.06, 1.06); ax.set_ylim(-0.06, S.VA[1] + 0.08)


def bottom_legend(fig, axes, D, sets):
    """shared corner colour key beneath each column (A=red top, B=blue, C=yellow)."""
    import textwrap
    for k, s in enumerate(sets):
        pos = axes[-1, k].get_position()
        xc, y = pos.x0 + pos.width / 2, pos.y0 - 0.012
        for nm, col in zip(D[s]["names"], S.VCOL):
            lines = textwrap.wrap(str(nm), 34) or [""]
            fig.text(xc, y, "■ " + lines[0], color=col, ha="center", va="top", fontsize=6.5, weight="bold")
            y -= 0.020
            for cont in lines[1:]:
                fig.text(xc, y, "   " + cont, color=col, ha="center", va="top", fontsize=6.5, weight="bold")
                y -= 0.020
            y -= 0.006


def barys_xy():
    barys = [(a, b, S.RES - a - b) for a in range(S.RES + 1) for b in range(S.RES + 1 - a)]
    return barys, np.array([S.bary_to_xy(b) for b in barys])


def plot_rgb(D, sets, out):
    rows = ["ideal"] + METHODS
    barys, P = barys_xy()
    fig, axes = plt.subplots(len(rows), len(sets), figsize=(1.85 * len(sets) + 0.8, 1.55 * len(rows) + 0.5), squeeze=False)
    for i, ch in enumerate(rows):
        for k, s in enumerate(sets):
            ax = axes[i, k]; cols = []
            for b in barys:
                if ch == "ideal":
                    w = np.array(b, float) / S.RES
                else:
                    w = D[s]["cellw"].get(f"{ch}|{'-'.join(map(str,b))}")
                    w = np.array(w) if w is not None else None
                cols.append((1, 1, 1, 0) if w is None else S.rgb_blend(sharpen(w), whiten=WHITEN))
            sca = ax.scatter(P[:, 0], P[:, 1], c=cols, s=CELL_SIZE, marker="h", edgecolors=CELL_EDGE, linewidths=CELL_EDGE_LW, zorder=3)
            sca.set_clip_path(plt.Polygon([S.VA, S.VB, S.VC], closed=True, transform=ax.transData))
            frame(ax)
            if i == 0:
                ax.set_title(str(s).replace("_mild", ""), fontsize=10.5, pad=8)
    for i, ch in enumerate(rows):
        pos = axes[i, 0].get_position()
        fig.text(0.02, pos.y0 + pos.height / 2, LAB[ch], rotation=90, va="center", ha="center", fontsize=9.5, weight="bold")
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.12, top=0.92, wspace=0.06, hspace=0.12)
    bottom_legend(fig, axes, D, sets)
    for ext in ("pdf",):
        fig.savefig(os.path.join(HERE, "figs", out + "." + ext), bbox_inches="tight", pad_inches=0.02, transparent=(ext == "pdf"), dpi=300)
    plt.close(fig); print("saved figs/%s.png" % out)


def plot_copy(D, sets, out):
    barys, P = barys_xy()
    fig, axes = plt.subplots(len(METHODS), len(sets), figsize=(1.85 * len(sets) + 1.05, 1.55 * len(METHODS) + 0.5), squeeze=False)
    allv = [v for s in sets for v in D[s]["copy"].values() if v is not None]
    lo, hi = float(np.percentile(allv, 2)), float(np.percentile(allv, 98))
    sca = None
    for i, ch in enumerate(METHODS):
        for k, s in enumerate(sets):
            ax = axes[i, k]; vals, pts = [], []
            for b, p in zip(barys, P):
                v = D[s]["copy"].get(f"{ch}|{'-'.join(map(str,b))}")
                if v is None:
                    continue
                pts.append(p); vals.append(v)
            pts = np.array(pts)
            sca = ax.scatter(pts[:, 0], pts[:, 1], c=vals, cmap="cividis", vmin=lo, vmax=hi,
                             s=CELL_SIZE, marker="h", edgecolors=CELL_EDGE, linewidths=CELL_EDGE_LW, zorder=3)
            sca.set_clip_path(plt.Polygon([S.VA, S.VB, S.VC], closed=True, transform=ax.transData))
            frame(ax)
            if i == 0:
                ax.set_title(str(s).replace("_mild", ""), fontsize=10.5, pad=8)
    for i, ch in enumerate(METHODS):
        pos = axes[i, 0].get_position()
        fig.text(0.045, pos.y0 + pos.height / 2, LAB[ch], rotation=90, va="center", ha="center", fontsize=10.5, weight="bold")
    fig.subplots_adjust(left=0.09, right=0.9, bottom=0.14, top=0.92, wspace=0.06, hspace=0.12)
    bottom_legend(fig, axes, D, sets)
    cax = fig.add_axes([0.915, 0.24, 0.017, 0.54]); cb = fig.colorbar(sca, cax=cax)
    cb.set_label(f"abstract verbatim copy-rate ({lo:.2f}-{hi:.2f})", fontsize=11)
    for ext in ("pdf",):
        fig.savefig(os.path.join(HERE, "figs", out + "." + ext), bbox_inches="tight", pad_inches=0.02, transparent=(ext == "pdf"), dpi=300)
    plt.close(fig); print("saved figs/%s.png" % out)


import seaborn as sns
import colorsys
_CB = sns.color_palette("colorblind")   # no green (index 2) used below


def _mute(c, f=0.35, l=1.18):
    """desaturate + lighten a colour so it recedes (used for the baselines)."""
    h, li, s = colorsys.rgb_to_hls(*c[:3])
    return colorsys.hls_to_rgb(h, min(1, li * l), max(0, s * f))


MCOL = {"doc2lora": (0.90, 0.06, 0.06),          # bright red — emphasised
        "incontext": _mute(_CB[0]),               # muted blue   — baseline
        "icae": _mute(_CB[1])}                     # muted orange — baseline


def plot_curve(D, sets, out):
    """fusion smoothness: measured corner-weight vs ideal barycentric weight.
    diagonal = smooth proportional fusion; step = winner-take-all; flat = confabulate."""
    bins = np.linspace(0, 1, 11)
    ctr = 0.5 * (bins[1:] + bins[:-1])
    fig, axes = plt.subplots(1, len(sets), figsize=(3.2 * len(sets) + 0.4, 3.1), squeeze=False)
    for k, s in enumerate(sets):
        ax = axes[0, k]
        ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1, zorder=1)
        for ch in METHODS:
            xs, ys = [], []
            for key, w in D[s]["cellw"].items():
                if not key.startswith(ch + "|") or w is None:
                    continue
                b = [int(x) for x in key.split("|")[1].split("-")]
                tot = sum(b)
                for j in range(3):
                    xs.append(b[j] / tot); ys.append(w[j])
            xs, ys = np.array(xs), np.array(ys)
            r = float(np.corrcoef(xs, ys)[0, 1])
            mean = [ys[(xs >= bins[i]) & (xs < bins[i + 1] + (i == len(bins) - 2) * 1e-9)].mean()
                    if ((xs >= bins[i]) & (xs < bins[i + 1] + 1e-9)).any() else np.nan
                    for i in range(len(bins) - 1)]
            ax.plot(ctr, mean, "-o", color=MCOL[ch], ms=3.5, lw=1.6,
                    label=LAB[ch], zorder=3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel("ideal corner weight");
        if k == 0:
            ax.set_ylabel("measured corner weight")
        ax.set_title(str(s).replace("_mild", ""), fontsize=10.5)
        ax.legend(fontsize=15, loc="upper left", frameon=False)
    fig.tight_layout()
    for ext in ("pdf",):
        fig.savefig(os.path.join(HERE, "figs", out + "." + ext), bbox_inches="tight", pad_inches=0.03, transparent=(ext == "pdf"), dpi=300)
    plt.close(fig); print("saved figs/%s.png" % out)


def split_kw5(s):
    out = []
    for x in re.split(r"[,\n;]+", s or ""):
        x = re.sub(r"^\s*[-*\d.)]+\s*", "", x).strip()
        if 2 <= len(x) <= 50 and not x.lower().startswith(("here", "keyword", "list")):
            out.append(x)
    return out[:5]


def plot_keywords(sets, out, method):
    """decoded keywords at the 3 pair-midpoints (A+B, A+C, B+C) and the centre."""
    import textwrap
    FUS = [((6, 6, 0), "A+B"), ((6, 0, 6), "A+C"), ((0, 6, 6), "B+C"), ((4, 4, 4), "centre")]
    cen = S.bary_to_xy((4, 4, 4))
    fig, axes = plt.subplots(1, len(sets), figsize=(4.6 * len(sets) + 0.3, 4.7), squeeze=False)
    for k, s in enumerate(sets):
        ax = axes[0, k]
        base = json.load(open(os.path.join(sc.RESULTS, f"absfollow_{s}{SUF}.json")))["cells"]
        if method == "icae":
            icp = os.path.join(sc.RESULTS, f"absfollow_{s}{SUF}_icae.json")
            cells = json.load(open(icp))["cells"]
            kwmap = {tuple(c["bary"]): (c.get("icae") or {}).get("kw", "") for c in cells}
        else:
            kwmap = {tuple(c["bary"]): (c.get(method) or {}).get("kw", "") for c in base}
        spec = sc.load_corners(sc.corners_path(s))
        names = [str(spec["names"][key]) for key in spec["keys"]]
        for v0, v1 in [(S.VA, S.VB), (S.VB, S.VC), (S.VC, S.VA)]:
            ax.plot([v0[0], v1[0]], [v0[1], v1[1]], "-", color="0.35", lw=0.8, zorder=1)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(-0.30, 1.30); ax.set_ylim(-0.32, 1.12)
        # corner names outside the vertices
        for nm, v, col, ha, dy in zip(names, [S.VA, S.VB, S.VC], S.VCOL,
                                      ["center", "left", "right"], [0.05, -0.05, -0.05]):
            dx = 0.06 if ha == "left" else (-0.06 if ha == "right" else 0)
            ax.text(v[0] + dx, v[1] + dy, "\n".join(textwrap.wrap(nm, 24)),
                    ha=ha, va=("bottom" if dy > 0 else "top"), fontsize=6, color=col, weight="bold", zorder=6)
        # keyword boxes: midpoints pushed outward from centroid so they don't collide
        for bary, lab in FUS:
            xy = S.bary_to_xy(bary)
            push = 1.0 if lab == "centre" else 1.55
            bx = cen + push * (xy - cen)
            kw = split_kw5(kwmap.get(bary, ""))
            ax.annotate("\n".join(kw), xy=xy, xytext=bx, ha="center", va="center", fontsize=6,
                        color="0.1", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=str(0.55), lw=0.6, alpha=0.95),
                        arrowprops=dict(arrowstyle="-", color="0.6", lw=0.6))
        ax.text(0.5, -0.28, str(s).replace("_mild", ""), transform=ax.transData,
                ha="center", va="center", fontsize=11, weight="bold")
    pass  # no paper title
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("pdf",):
        fig.savefig(os.path.join(HERE, "figs", out + "." + ext), bbox_inches="tight", pad_inches=0.03, transparent=(ext == "pdf"), dpi=300)
    plt.close(fig); print("saved figs/%s.png" % out)


def main(argv):
    setup_style(font_scale=0.9)
    sets = [a for a in argv if not a.startswith("--")] or ["strat00", "strat80"]
    D = json.load(open(os.path.join(sc.RESULTS, f"absf_metrics{SUF}.json"))) if "--replot" in argv else compute(sets)
    plot_rgb(D, sets, "absf_rgb")
    plot_copy(D, sets, "absf_copyrate")
    plot_curve(D, sets, "absf_fusion_curve")
    for mth in METHODS:
        plot_keywords(sets, f"absf_keywords_{mth}", mth)
    for s in sets:
        for ch in METHODS:
            cv = [v for k, v in D[s]["copy"].items() if k.startswith(ch + "|") and v is not None]
            print(f"  {s:9} {LAB[ch]:<11} copy={np.mean(cv):.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
