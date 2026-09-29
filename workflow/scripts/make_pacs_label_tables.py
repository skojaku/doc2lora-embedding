"""Generate LaTeX tables pairing decoded Doc2LoRA labels with the actual PACS code.

Two outputs:
  - pacs_labels_main.tex  : a representative sample for the main text
  - pacs_labels_si.tex     : the full set of decoded PACS nodes for the SI

The decoded labels are the text-free Doc2LoRA-qwen full-rank outputs reported in
RESULTS.md.  Ground-truth PACS descriptions and paper counts come from the APS
groups table; only nodes that carry a real PACS code (divisions + subdivisions,
plus the PACS top-level fields) are shown.

Run: python make_pacs_label_tables.py
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
from bench_data import out_dir   # bench_data.py sits next to this file
DATA = out_dir("labels")           # where this chain writes

GROUPS = (
    out_dir("pacs") / "results" / "groups.parquet"   # the labelled PACS tree (pacs_groups.smk)
)
RADIUS_CSV = DATA / "length_vs_breadth.csv"

# Text-free Doc2LoRA-qwen decoded labels: FULL-RANK node means
# (qwen_fullrank_field23.json, decoded from the [36,8,512] mean per node).
import json
DECODED = json.loads((DATA / "qwen_fullrank_field23.json").read_text())

# Order for the SI table: grouped by PACS field, parent before children.
SI_ORDER = [
    "0", "03", "03.65", "03.67", "05", "05.40", "05.45",
    "2", "42", "42.50", "42.65", "47", "47.20", "47.27",
    "3", "71", "71.10", "71.20", "75", "75.10", "75.30",
    "6", "11", "11.10", "11.15", "12", "12.38", "12.60",
]

# Representative spread for the main text.
MAIN_ORDER = [
    "03", "03.67", "05", "05.45", "42", "42.65", "47", "47.27", "75", "12.38",
]

LEVEL_NAME = {"main": "field", "division": "division", "subdivision": "subdiv."}


def load_meta():
    gl = pd.read_parquet(GROUPS)
    gl["group_id"] = gl["group_id"].astype(str)
    gt = {}
    level = {}
    for _, row in gl.iterrows():
        gt[row.group_id] = row.gt_label_text
        level[row.group_id] = row.level
    counts = {}
    rad = pd.read_csv(RADIUS_CSV)
    rad["node"] = rad["node"].astype(str)
    for _, row in rad.iterrows():
        counts[row.node] = int(row.n)
    return gt, level, counts


def fmt_count(n):
    return f"{n / 1000:.0f}k" if n >= 10_000 else f"{n / 1000:.1f}k"


def fmt_code(code):
    # Format PACS code in a fixed-width font; field-level ids get a dagger.
    return rf"\texttt{{{code}}}"


def build_rows(order, gt, level, counts):
    rows = []
    for code in order:
        rows.append(
            (
                fmt_code(code),
                LEVEL_NAME.get(level.get(code, ""), level.get(code, "")),
                gt.get(code, ""),
                DECODED[code],
                fmt_count(counts.get(code, 0)) if code in counts else "",
            )
        )
    return rows


def emit_table(rows, caption, label, longtable=False):
    lines = []
    colspec = "@{}llp{0.30\\textwidth}p{0.30\\textwidth}r@{}"
    header = (
        "PACS & level & Actual PACS description & "
        "Decoded Doc2LoRA label & $n$ \\\\"
    )
    if longtable:
        lines.append(f"\\begin{{longtable}}{{{colspec}}}")
        lines.append(f"\\caption{{{caption}}}\\label{{{label}}}\\\\")
        lines.append("\\toprule")
        lines.append(header)
        lines.append("\\midrule\\endfirsthead")
        lines.append("\\toprule")
        lines.append(header)
        lines.append("\\midrule\\endhead")
        lines.append("\\bottomrule\\endfoot")
    else:
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\small")
        lines.append(f"\\caption{{{caption}}}")
        lines.append(f"\\label{{{label}}}")
        lines.append(f"\\begin{{tabular}}{{{colspec}}}")
        lines.append("\\toprule")
        lines.append(header)
        lines.append("\\midrule")
    for code, lvl, gt_desc, decoded, n in rows:
        lines.append(f"{code} & {lvl} & {gt_desc} & {decoded} & {n} \\\\")
    if longtable:
        lines.append("\\end{longtable}")
    else:
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def main():
    gt, level, counts = load_meta()

    main_rows = build_rows(MAIN_ORDER, gt, level, counts)
    si_rows = build_rows(SI_ORDER, gt, level, counts)

    main_caption = (
        "Decoded Doc2LoRA labels versus the actual PACS descriptions for a "
        "representative sample of PACS nodes. Labels are produced text-free by "
        "decoding the full-rank Doc2LoRA mean of each node; $n$ is the number "
        "of papers. The full set is given in the SI (Table~\\ref{tab:pacs-labels-si})."
    )
    si_caption = (
        "Decoded Doc2LoRA labels versus the actual PACS descriptions for every "
        "decoded PACS node (fields, divisions, and subdivisions). Labels are "
        "produced text-free by decoding the full-rank Doc2LoRA mean of each "
        "node; $n$ is the number of papers."
    )

    main_tex = emit_table(
        main_rows, main_caption, "tab:pacs-labels-main", longtable=False
    )
    si_tex = emit_table(si_rows, si_caption, "tab:pacs-labels-si", longtable=True)

    (DATA / "pacs_labels_main.tex").write_text(main_tex)
    (DATA / "pacs_labels_si.tex").write_text(si_tex)
    print("wrote pacs_labels_main.tex")
    print("wrote pacs_labels_si.tex")
    print()
    print(main_tex)


if __name__ == "__main__":
    main()
