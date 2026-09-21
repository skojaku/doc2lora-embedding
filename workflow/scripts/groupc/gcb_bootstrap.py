"""[CPU] Bootstrap the Group C similarity pools into one table (#69 / #93 / #72).

Metrics mirror the manuscript's: collab AUC averaged over windows with the candidate pair as the
bootstrap unit, next-paper AUC with the QUERY as the unit (cluster bootstrap), topic macro-F1 over
test papers.  The `topic_temporal` pool adds the train<=cutoff / test>cutoff robustness row, and the
latest next-paper window is reported separately as the late-period row.

AUC is computed with the rank identity rather than sklearn inside the bootstrap loop, so 500
replicates over a 300k-row pool take seconds instead of hours.

Out: gcb_summary.csv + figs/groupc_similarity.tex (#69/#93) + figs/groupc_temporal.tex (#72)
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

NBOOT = int(snakemake.params.nboot)               # noqa: F821
POOLS = snakemake.params.pool_dir                 # noqa: F821
FIELDS = list(snakemake.params.fields)            # noqa: F821
rng = np.random.default_rng(0)


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUC via ranks (ties averaged)."""
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within tied score groups
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ci(reps):
    reps = [r for r in reps if np.isfinite(r)]
    if not reps:
        return (np.nan, np.nan)
    return (float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5)))


def boot_collab(df, col, nboot):
    """AUC per window, averaged; bootstrap resamples candidate pairs within each window."""
    d = df.dropna(subset=[col])
    wins = [g for _, g in d.groupby("window") if g.y.nunique() > 1]
    if not wins:
        return np.nan, (np.nan, np.nan)
    ys = [g.y.values.astype(np.int8) for g in wins]
    ss = [g[col].values.astype(np.float64) for g in wins]
    point = float(np.mean([auc(y, s) for y, s in zip(ys, ss)]))
    reps = []
    for _ in range(nboot):
        vals = []
        for y, s in zip(ys, ss):
            k = rng.integers(0, len(y), len(y))
            vals.append(auc(y[k], s[k]))
        reps.append(np.nanmean(vals))
    return point, ci(reps)


def boot_np(df, col, nboot, windows=None):
    """Pooled AUC; bootstrap resamples QUERIES (clusters of candidate rows)."""
    d = df if windows is None else df[df.window.isin(windows)]
    d = d.dropna(subset=[col])
    if d.is_pos.nunique() < 2:
        return np.nan, (np.nan, np.nan)
    y = d.is_pos.values.astype(np.int8)
    s = d[col].values.astype(np.float64)
    point = auc(y, s)
    q = d.query_idx.values
    order = np.argsort(q, kind="stable")
    qs = q[order]
    bounds = np.searchsorted(qs, np.unique(qs), side="left")
    ends = np.append(bounds[1:], len(qs))
    groups = [order[a:b] for a, b in zip(bounds, ends)]
    reps = []
    for _ in range(nboot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        reps.append(auc(y[idx], s[idx]))
    return point, ci(reps)


def boot_f1(df, col, nboot):
    yt = df.true.values
    yp = df[col].values
    point = float(f1_score(yt, yp, average="macro"))
    reps = []
    for _ in range(max(nboot // 4, 50)):
        k = rng.integers(0, len(yt), len(yt))
        reps.append(f1_score(yt[k], yp[k], average="macro"))
    return point, ci(reps)


SKIP = {"window", "a", "b", "y", "query_idx", "cand_pid", "is_pos", "paper_id", "true", "year"}
rows = []
for field in FIELDS:
    for task in ("collab", "np", "topic", "topic_temporal"):
        path = f"{POOLS}/{task}_{field}.parquet"
        if not os.path.exists(path):
            print(f"  (missing {path})", flush=True)
            continue
        df = pd.read_parquet(path)
        cols = [c for c in df.columns if c not in SKIP]
        for col in cols:
            if task == "collab":
                pt, c = boot_collab(df, col, NBOOT)
            elif task == "np":
                pt, c = boot_np(df, col, NBOOT)
            else:
                pt, c = boot_f1(df, col, NBOOT)
            rows.append({"field": field, "task": task, "method": col, "value": pt,
                         "lo": c[0], "hi": c[1], "n": len(df)})
            print(f"  {field:11} {task:15} {col:28} {pt:.4f} [{c[0]:.4f}, {c[1]:.4f}]", flush=True)
        if task == "np":
            late = sorted(df.window.unique())[-1]
            for col in cols:
                pt, c = boot_np(df, col, NBOOT, windows=[late])
                rows.append({"field": field, "task": f"np_late", "method": col, "value": pt,
                             "lo": c[0], "hi": c[1], "n": int((df.window == late).sum()),
                             "window": int(late)})

S = pd.DataFrame(rows)
os.makedirs(os.path.dirname(snakemake.output.csv), exist_ok=True)      # noqa: F821
S.to_csv(snakemake.output.csv, index=False)                            # noqa: F821

NICE = {"gene": "Doc2LoRA (raw)", "genkron": "Doc2LoRA $+g_\\theta$ (reported)",
        "gene_kron_gc": "Doc2LoRA $+g_\\theta$ (re-trained here)",
        "gene_kron_gc_pre2018": "Doc2LoRA $+g_\\theta$ (pre-2018 citations only)",
        "sbert": "SBERT all-mpnet", "sbert_kron_gc": "\\quad $+g_\\theta$",
        "sbert_kron_gc_pre2018": "\\quad $+g_\\theta$ (pre-2018)",
        "specter2": "SPECTER2", "specter2_kron_gc": "\\quad $+g_\\theta$",
        "icae": "ICAE (mean slot)",
        "instructor": "Instructor", "instructor_kron_gc": "\\quad $+g_\\theta$",
        "embeddinggemma": "EmbeddingGemma", "embeddinggemma_kron_gc": "\\quad $+g_\\theta$",
        "bge": "BGE-large-en-v1.5 (2023)", "bge_kron_gc": "\\quad $+g_\\theta$",
        "gte": "GTE-multilingual-base (2024)", "gte_kron_gc": "\\quad $+g_\\theta$",
        "e5mistral": "E5-Mistral-7B-instruct", "e5mistral_kron_gc": "\\quad $+g_\\theta$"}
TASKS = [("collab", "collab AUC"), ("np", "next-paper AUC"), ("topic", "topic macro-F1")]


def fmt(v, lo, hi):
    if not np.isfinite(v):
        return "---"
    return f"{v:.3f}\\,{{\\tiny[{lo:.3f},{hi:.3f}]}}"


def cell(field, task, method):
    r = S[(S.field == field) & (S.task == task) & (S.method == method)]
    return fmt(r.value.iloc[0], r.lo.iloc[0], r.hi.iloc[0]) if len(r) else "---"


with open(snakemake.output.table, "w") as fh:                          # noqa: F821
    fh.write("% auto-generated by workflow/scripts/groupc/gcb_bootstrap.py (#69/#93) -- do not edit\n")
    fh.write("\\begin{tabular}{l" + "r" * (len(FIELDS) * len(TASKS)) + "}\n\\toprule\n")
    fh.write("method" + "".join(f" & \\multicolumn{{{len(TASKS)}}}{{c}}{{{f}}}" for f in FIELDS) + " \\\\\n")
    fh.write("".join(" & " + t[1] for _ in FIELDS for t in TASKS) + " \\\\\n\\midrule\n")
    for m in [k for k in NICE if "pre2018" not in k]:
        if m not in set(S.method):
            continue
        fh.write(NICE[m] + " & " +
                 " & ".join(cell(f, tk, m) for f in FIELDS for tk, _ in TASKS) + " \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

with open(snakemake.output.temporal_table, "w") as fh:                 # noqa: F821
    fh.write("% auto-generated by workflow/scripts/groupc/gcb_bootstrap.py (#72) -- do not edit\n")
    fh.write("\\begin{tabular}{llrrr}\n\\toprule\n")
    fh.write("field & transform & topic F1 (random split) & topic F1 (temporal split) "
             "& next-paper AUC (last window) \\\\\n\\midrule\n")
    tm = [m for m in set(S.method) if m in NICE and ("pre2018" in m or m in
          ("gene", "genkron", "gene_kron_gc", "sbert", "sbert_kron_gc"))]
    for field in FIELDS:
        for m in sorted(tm):
            fh.write(f"{field} & {NICE.get(m, m)} & {cell(field, 'topic', m)} & "
                     f"{cell(field, 'topic_temporal', m)} & {cell(field, 'np_late', m)} \\\\\n")
        fh.write("\\midrule\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
print(f"[gcb-bootstrap] wrote {snakemake.output.csv}, {snakemake.output.table}, "   # noqa: F821
      f"{snakemake.output.temporal_table}", flush=True)                             # noqa: F821
