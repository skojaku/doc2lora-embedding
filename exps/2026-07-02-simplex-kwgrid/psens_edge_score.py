"""Is the §4.3 edge result a property of the decode PROMPT?

The paper decodes every interpolated point under one instruction ("Write a detailed
abstract ... problem, methods, and findings"). Appendix app:prompt-sensitivity already
reports a paraphrase sweep, but over a different prompt family (2-3 sentence "combined
research idea"), so the two claims §4.3 actually makes -- Doc2LoRA/ICAE track the
barycentric weight while in-context sits at the midpoint, and copy rate orders
ICAE > in-context > Doc2LoRA -- have never been checked against paraphrase.

This scores the same two curves under every paraphrase in psens_prompts.ABS_PROMPTS
(index 0 = the manuscript's wording) and asks whether the ORDERING and SHAPE survive,
not whether the numbers are identical.

Metric definitions are the ones the paper's own panels use:
  landing position  actual_t = (d - A).(B - A) / |B - A|^2   (pair_axis_metrics.py)
  copy rate         fraction of decoded word-tokens present in either source lead,
                    tokens = [a-z][a-z-]{2,}                 (pair_axis_copyrate.py)

  python psens_edge_score.py --pairs 20 --prompts 0 1 2 3
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simplex_common as sc                      # noqa: E402
from pair_axis_copyrate import toks              # noqa: E402  (same token rule as the paper panel)
import psens_prompts as pp                       # noqa: E402

METHODS = ["doc2lora", "incontext", "icae"]
LAB = {"doc2lora": "\\doctolora", "incontext": "in-context", "icae": "\\texttt{ICAE}"}
PLAB = {"doc2lora": "D2L", "incontext": "In-ctx", "icae": "ICAE"}
# Same three arms as the Figure 2 mixing panels, so the appendix figure reads as
# the same experiment. Values copied from workflow/plot/_fig2_palette.py
# (LINE_COLOR/MLAB) rather than imported: this script runs in the GPU sandbox and
# must not grow a dependency on the workflow tree. Keep the two in step.
INK, INK2 = "#0b0b0b", "#52514e"                 # text / axis tokens
SURFACE = "#ffffff"                              # the ring around every marker
MCOL = {"doc2lora": "#eb6834", "icae": "#3d3c39", "incontext": "#8b8a85"}
GUIDE = "#a8a7a1"                                # GREY_MARK2: the ideal diagonal

# Fig. 2 is drawn on a 16.4in canvas and included at \linewidth (5.5in), so every
# size in fig_pacs_clustering.py lands on the page at 5.5/16.4 = .335 of its
# value. This figure is drawn at 5.5in, so the same on-page result means using
# those numbers already scaled. Keep this table in step with panels (e)/(f).
SCALE = 5.5 / 16.4
FS_TICK, FS_LABEL, FS_LEGEND = 20 * SCALE, 24 * SCALE, 20 * SCALE
LW_MAIN, LW_REST = 3.2 * SCALE, 2.4 * SCALE      # doc2lora is the heavier arm
LW_GUIDE, MS, MEW = 1.4 * SCALE, 5.5 * SCALE, 0.8 * SCALE
STRATA = ["L1", "L5"]
BASE_SUF = "_pairaxis"


def suffix(p):
    return BASE_SUF if int(p) == 0 else f"{BASE_SUF}_p{int(p)}"


def load_cells(set_name, p):
    """{bary -> {method -> abstract}} for one pair under one prompt."""
    suf = suffix(p)
    out = {}
    base = os.path.join(sc.RESULTS, f"absfollow_{set_name}{suf}.json")
    icae = os.path.join(sc.RESULTS, f"absfollow_{set_name}{suf}_icae.json")
    if os.path.exists(base):
        for c in json.load(open(base))["cells"]:
            for ch in ("doc2lora", "incontext"):
                if c.get(ch, {}).get("abs"):
                    out.setdefault(tuple(c["bary"]), {})[ch] = c[ch]["abs"]
    if os.path.exists(icae):
        for c in json.load(open(icae))["cells"]:
            if c.get("icae", {}).get("abs"):
                out.setdefault(tuple(c["bary"]), {})["icae"] = c["icae"]["abs"]
    return out


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def collect(pairs, prompts):
    """One pass over every (pair, prompt, cell, method) decode. Returns records plus the
    text list to embed, so SBERT is called once in a batch instead of per decode."""
    recs, texts = [], []
    for L in STRATA:
        for i in range(pairs):
            s = f"pair{L}_{i:02d}"
            cpath = sc.corners_path(s)
            if not os.path.exists(cpath):
                continue
            spec = sc.load_corners(cpath)
            srctok = set(toks(spec["leads"]["A"])) | set(toks(spec["leads"]["B"]))
            corner_idx = {}
            for k in ("A", "B"):
                corner_idx[k] = len(texts)
                texts.append(spec["leads"][k])
            for p in prompts:
                cells = load_cells(s, p)
                for bary, per_ch in cells.items():
                    ideal_t = bary[1] / sc.GRID_DEN
                    for ch, ab in per_ch.items():
                        at = toks(ab)
                        if len(at) < 6:
                            continue
                        recs.append({"set": s, "stratum": L, "prompt": int(p), "method": ch,
                                     "ideal_t": ideal_t, "bary": list(bary),
                                     "copy": sum(w in srctok for w in at) / len(at),
                                     "idx": len(texts), "A_idx": corner_idx["A"],
                                     "B_idx": corner_idx["B"]})
                        texts.append(ab)
    return recs, texts


def embed(texts):
    from sentence_transformers import SentenceTransformer
    sb = SentenceTransformer("all-mpnet-base-v2",
                             device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    E = sb.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    return unit(np.asarray(E))


def add_landing(recs, E):
    for r in recs:
        A, B = E[r["A_idx"]], E[r["B_idx"]]
        axis = B - A
        r["actual_t"] = float((E[r["idx"]] - A) @ axis / (float(axis @ axis) + 1e-12))


def boot_ci(v, B=2000, seed=0):
    v = np.asarray(v, dtype=float)
    if not len(v):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def curve(recs, ts, key):
    """mean of `key` at each of the 13 ideal weights."""
    out = []
    for t in ts:
        v = [r[key] for r in recs if abs(r["ideal_t"] - t) < 1e-9]
        out.append(float(np.mean(v)) if v else float("nan"))
    return out


def track_fit(recs):
    """OLS slope and Pearson r of landing position on requested weight. A method that
    tracks the mixing weight has slope ~1; one that sits at the midpoint has slope ~0."""
    if len(recs) < 3:
        return float("nan"), float("nan")
    x = np.array([r["ideal_t"] for r in recs])
    y = np.array([r["actual_t"] for r in recs])
    slope = float(np.polyfit(x, y, 1)[0])
    r = float(np.corrcoef(x, y)[0, 1])
    return slope, r


def cross_prompt(recs, E, prompts):
    """Same point, different prompt: how similar are the decodes? The reference scale is
    the similarity between decodes of DIFFERENT points under one fixed prompt -- without
    it a cosine of .7 means nothing."""
    by_key = {}
    for r in recs:
        by_key.setdefault((r["set"], r["method"], tuple(r["bary"])), {})[r["prompt"]] = r["idx"]
    same = {ch: [] for ch in METHODS}
    for (s, ch, b), d in by_key.items():
        for p1, p2 in itertools.combinations(sorted(d), 2):
            same[ch].append(float(E[d[p1]] @ E[d[p2]]))
    # null: different (set, cell) pairs under the manuscript prompt only
    p0 = [r for r in recs if r["prompt"] == min(prompts)]
    null = {ch: [] for ch in METHODS}
    rng = np.random.default_rng(0)
    for ch in METHODS:
        idx = [r["idx"] for r in p0 if r["method"] == ch]
        if len(idx) < 2:
            continue
        a = rng.choice(idx, size=min(4000, len(idx) * 4))
        b = rng.choice(idx, size=len(a))
        keep = a != b
        null[ch] = [float(E[i] @ E[j]) for i, j in zip(a[keep], b[keep])]
    return ({ch: float(np.mean(v)) if v else float("nan") for ch, v in same.items()},
            {ch: float(np.mean(v)) if v else float("nan") for ch, v in null.items()},
            {ch: float(np.min(v)) if v else float("nan") for ch, v in same.items()})


def write_tex(path, summary, prompts):
    rows = []
    for ch in METHODS:
        for p in prompts:
            d = summary[f"{ch}|{p}"]
            tag = "reported" if p == 0 else f"paraphrase {p}"
            rows.append(
                f"{LAB[ch]} & {tag} & {d['slope_L1']:.2f} & {d['slope_L5']:.2f} & "
                f"{d['copy_L1'][0]:.2f}--{d['copy_L1'][1]:.2f} & "
                f"{d['copy_L5'][0]:.2f}--{d['copy_L5'][1]:.2f} \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    body = "\n".join(rows)
    tex = f"""% auto-generated by exps/2026-07-02-simplex-kwgrid/psens_edge_score.py -- do not edit
\\begin{{tabular}}{{llrrrr}}
\\toprule
 &  & \\multicolumn{{2}}{{c}}{{tracking slope}} & \\multicolumn{{2}}{{c}}{{copy rate (range over the edge)}} \\\\
\\cmidrule(lr){{3-4}} \\cmidrule(lr){{5-6}}
method & decode prompt & near & far & near & far \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
"""
    open(path, "w").write(tex)


def plot(path, summary, prompts, ts):
    """Figure 3, drawn as panels (e)/(f) of Figure 2 with a paraphrase sweep on top.

    Same arms, same colours (workflow/plot/_fig2_palette.py LINE_COLOR/MLAB),
    same axis wording, same ideal diagonal, and the same line weights and white
    marker rings once Figure 2's canvas scaling is taken out -- so the two
    figures read as one experiment. What is new here is the dashed paraphrase
    curves behind each solid arm.

    Curves come from the summary dict, which is what lands in the JSON, so the
    figure redraws without re-running the decodes (--replot).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.facecolor": "white", "figure.facecolor": "white",
    })

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.6), sharex=True, sharey="row")
    panels = [("landing_curve", "Actual mixing"), ("copy_curve", "Copy rate")]
    reported = min(prompts)

    for col, L in enumerate(STRATA):
        for row, (key, ylab) in enumerate(panels):
            ax = axes[row][col]
            if row == 0:
                ax.plot([0, 1], [0, 1], "--", color=GUIDE, lw=LW_GUIDE, zorder=1)
                ax.text(0.22, 0.25, "ideal", rotation=45, ha="center", va="bottom",
                        fontsize=FS_TICK - 0.7, color=INK2, rotation_mode="anchor")
            handles = {}
            for ch in ["doc2lora", "icae", "incontext"]:
                z = 6 if ch == "doc2lora" else 3
                for p in prompts:
                    y = summary.get(f"{ch}|{p}", {}).get(f"{key}_{L}")
                    if not y:
                        continue
                    if p == reported:
                        handles[ch] = ax.plot(
                            ts, y, "-o", color=MCOL[ch], ms=MS, mec=SURFACE, mew=MEW,
                            lw=LW_MAIN if ch == "doc2lora" else LW_REST, zorder=z)[0]
                        handles[ch].end = y[-1]
                    else:
                        ax.plot(ts, y, "--", color=MCOL[ch], lw=LW_REST * 0.55,
                                alpha=0.8, zorder=z - 1)
            if row == 0 and col == 0:                  # arms ordered as they end
                order = sorted(handles, key=lambda c: -handles[c].end)
                ax.legend([handles[c] for c in order], [PLAB[c] for c in order],
                          loc="upper left", frameon=False, fontsize=FS_LEGEND,
                          handlelength=1.8, borderaxespad=0.1, labelspacing=0.25,
                          handletextpad=0.5)
            if row == 0:
                ax.set_title("near (L1)" if L == "L1" else "far (L5)",
                             fontsize=FS_LABEL, color=INK, pad=3)
            if row == 1:
                ax.set_xlabel("Ideal mixing", fontsize=FS_LABEL)
            if col == 0:
                ax.set_ylabel(ylab, fontsize=FS_LABEL)
            ax.set_xlim(-0.03, 1.03)
            ax.set_xticks([0, 0.5, 1.0])
            ax.set_xticklabels(["0", ".5", "1"])
            ax.tick_params(labelsize=FS_TICK, length=2.0, width=0.5, pad=1.5)
            for side in ("bottom", "left"):
                ax.spines[side].set_linewidth(0.6)

    fig.tight_layout(pad=0.25, h_pad=0.5, w_pad=0.8)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.01, dpi=300, transparent=True)
    plt.close(fig)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--prompts", nargs="+", type=int, default=list(range(pp.n_prompts())))
    ap.add_argument("--json", default=os.path.join(sc.RESULTS, "psens_edge.json"))
    ap.add_argument("--tex", default="figs/prompt_sensitivity_edge.tex")
    ap.add_argument("--fig", default="figs/psens_edge_curves.pdf")
    ap.add_argument("--replot", action="store_true",
                    help="redraw the figure (and table) from an existing --json, "
                         "without re-running the decodes or the embedder")
    a = ap.parse_args()

    if a.replot:
        prev = json.load(open(a.json))
        write_tex(a.tex, prev["summary"], prev["prompts"])
        print(f"wrote {a.tex}")
        plot(a.fig, prev["summary"], prev["prompts"], prev["ideal_t"])
        return

    recs, texts = collect(a.pairs, a.prompts)
    print(f"[collect] {len(recs)} decodes, {len(texts)} strings to embed", flush=True)
    E = embed(texts)
    add_landing(recs, E)

    ts = [k / sc.GRID_DEN for k in range(sc.GRID_DEN + 1)]
    summary = {}
    for ch in METHODS:
        for p in a.prompts:
            d = {}
            for L in STRATA:
                sub = [r for r in recs if r["method"] == ch and r["prompt"] == p
                       and r["stratum"] == L]
                slope, rr = track_fit(sub)
                cc = curve(sub, ts, "copy")
                lc = curve(sub, ts, "actual_t")
                d[f"slope_{L}"] = slope
                d[f"r_{L}"] = rr
                d[f"copy_{L}"] = [float(np.nanmin(cc)), float(np.nanmax(cc))] if cc else [np.nan]*2
                d[f"copy_curve_{L}"] = cc
                d[f"landing_curve_{L}"] = lc
                d[f"n_{L}"] = len(sub)
            summary[f"{ch}|{p}"] = d

    same, null, worst = cross_prompt(recs, E, a.prompts)
    out = {"pairs_per_stratum": a.pairs, "prompts": a.prompts,
           "prompt_texts": [pp.abs_prompt(p) for p in a.prompts],
           "summary": summary, "cross_prompt_sim": same,
           "cross_prompt_worst": worst, "across_point_null": null,
           "ideal_t": ts}
    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    json.dump(out, open(a.json, "w"), indent=1)
    print(f"wrote {a.json}")

    write_tex(a.tex, summary, a.prompts)
    print(f"wrote {a.tex}")
    plot(a.fig, summary, a.prompts, ts)

    # console verdict: does the qualitative reading survive every paraphrase?
    for L in STRATA:
        print(f"\n== {L} ==")
        for p in a.prompts:
            sl = {ch: summary[f'{ch}|{p}'][f'slope_{L}'] for ch in METHODS}
            cp = {ch: np.mean(summary[f'{ch}|{p}'][f'copy_curve_{L}']) for ch in METHODS}
            order = " > ".join(sorted(cp, key=lambda c: -cp[c]))
            print(f" prompt {p}: slope " +
                  ", ".join(f"{ch}={sl[ch]:.2f}" for ch in METHODS) +
                  f" | copy order {order}")
    print("\ncross-prompt sim:", {k: round(v, 3) for k, v in same.items()},
          "| across-point null:", {k: round(v, 3) for k, v in null.items()})


if __name__ == "__main__":
    main()
