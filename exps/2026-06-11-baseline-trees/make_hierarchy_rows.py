"""Build paper/iclr2026/hierarchy_rows.tex from the SCORED labels.

The table used to be hand-assembled, and drifted: it showed naming-LLM outputs for
\texttt{ICAE} / \texttt{vec2text} that the evaluation never scored. Generating it
from label_eval_nodes.json -- the same file label_eval_metric1/3 read -- means the
qualitative table and the quantitative table can only ever show the same strings.

Long labels are truncated at a word boundary for the page; the evaluation scores
them in full.

Reads:  label_eval_nodes.json, nodes.json (for the PACS display names)
Writes: paper/iclr2026/hierarchy_rows.tex
Run:    python make_hierarchy_rows.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "paper" / "iclr2026" / "hierarchy_rows.tex"
METHODS = ["doc2lora", "keyllm", "icae", "vec2text"]
MAXCH = 58

# PACS display names, in table order: (code, short name, depth)
ROWS = [
    ("0", "General", 0), ("03", "Quantum mechanics, QFT", 1),
    ("03.65", "Quantum mechanics", 2), ("03.67", "Quantum information", 2),
    ("05", "Statistical physics", 1), ("05.40", "Fluctuations, noise", 2),
    ("05.45", "Nonlinear dynamics, chaos", 2),
    ("2", "Classical physics", 0), ("42", "Optics", 1),
    ("42.50", "Quantum optics", 2), ("42.65", "Nonlinear optics", 2),
    ("47", "Fluid dynamics", 1), ("47.20", "Flow instabilities", 2),
    ("47.27", "Turbulent flows", 2),
    ("3", "Condensed matter", 0), ("71", "Electronic structure", 1),
    ("71.10", "Many-electron systems", 2), ("71.20", "Bulk band structure", 2),
    ("75", "Magnetic materials", 1), ("75.10", "Magnetic ordering", 2),
    ("75.30", "Ordered magnets", 2),
    ("6", "Elementary particles", 0), ("11", "Fields and particles", 1),
    ("11.10", "Field theory", 2), ("11.15", "Gauge field theories", 2),
    ("12", "Particle systematics", 1), ("12.38", "Quantum chromodynamics", 2),
    ("12.60", "Beyond standard model", 2),
]
INDENT = {0: "", 1: r"\hspace{0.8em}", 2: r"\hspace{1.8em}"}


def tex(s):
    for a, b in [("\\", r"\textbackslash "), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde "), ("^", r"\textasciicircum ")]:
        s = s.replace(a, b)
    return s


def cell(s):
    s = " ".join(str(s or "").split())
    if not s:
        return "---"
    if len(s) <= MAXCH:
        return tex(s)
    return tex(s[:MAXCH].rsplit(" ", 1)[0].rstrip(",;.")) + r"\,\dots"


def main():
    dec = {n["code"]: n["decoded"] for n in
           json.loads((HERE / "label_eval_nodes.json").read_text())}
    lines, prev_depth = [], None
    for code, name, depth in ROWS:
        if depth == 0 and prev_depth is not None:
            lines.append(r"\addlinespace[2pt]")
        prev_depth = depth
        head = f"{INDENT[depth]}{code}~~{name}"
        if depth == 0:
            head = INDENT[depth] + r"\textbf{" + f"{code}~~{name}" + "}"
        cells = [cell(dec[code].get(m)) for m in METHODS]
        cells[0] = r"\textbf{" + cells[0] + "}"          # Doc2LoRA column in bold
        lines.append(" & ".join([head] + cells) + r" \\")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(ROWS)} rows)")


if __name__ == "__main__":
    main()
