"""[CPU] One symmetric-supervision table over all 14 similarity tests (#145).

Merges the two halves of the control -- the 9 field tests (gcb_summary.csv, written by
gcb_bootstrap.py) and the 5 S2AND tests (gcb_s2and_summary.csv, written by gcb_s2and_bootstrap.py)
-- into a single table in which every embedding space has had the SAME citation-trained bijective
transform applied to it, and into a tidy long CSV for reading the deltas off directly.

"The same transform" means the same recipe -- same OpenAlex citation tuples, same InfoNCE with
in-batch plus one easy and one hard negative, same 8,000 steps, same optimiser, same seed -- not the
same parameter count. The factorisation follows the space: the gene tensor is 36 layers x 512
channels and gets n_L + n_d^2 = 262,692 parameters, while a flat 768-d text space gets one 768x768
channel map (590,593) and a 1,024-d one gets 1,049,601. ICAE is the outlier: its space is 128 memory
tokens x 4,096 channels and its map (trained by the ICAE chain on the same citation tuples) carries
16,781,440 parameters. Every baseline therefore receives a LARGER map than Doc2LoRA, which makes the
control conservative rather than generous to Doc2LoRA.

Out: figs/groupc_similarity_symmetric.tex (with intervals, two stacked blocks),
     figs/symmetric_adapter_values.tex (one wide block, values only, for the appendix),
     data/groupc/bench/gcb_symmetric.csv
"""
import os

import numpy as np
import pandas as pd

FIELD_CSV = snakemake.input.fields                   # noqa: F821
S2AND_CSV = snakemake.input.s2and                    # noqa: F821
OUT_TEX = snakemake.output.table                     # noqa: F821
OUT_VALUES = snakemake.output.values_table           # noqa: F821
OUT_CSV = snakemake.output.csv                       # noqa: F821
FIELDS = list(snakemake.params.fields)               # noqa: F821
S2AND = list(snakemake.params.s2and)                 # noqa: F821

# S2AND names its gene columns differently (gene/genkron are shared, the rest are per-space).
ALIAS = {"specter": "specter2", "specter_kron_gc": "specter2_kron_gc"}

F = pd.read_csv(FIELD_CSV)
S = pd.read_csv(S2AND_CSV)
S["method"] = S.method.map(lambda m: ALIAS.get(m, m))
A = pd.concat([F[F.task.isin(["collab", "np", "topic"])], S], ignore_index=True)

PAIRS = [("gene", "gene_kron_gc", "\\doctolora (Qwen3-4B)"),
         ("sbert", "sbert_kron_gc", "SBERT all-mpnet"),
         ("specter2", "specter2_kron_gc", "SPECTER2 / SPECTER"),
         ("instructor", "instructor_kron_gc", "Instructor"),
         ("embeddinggemma", "embeddinggemma_kron_gc", "EmbeddingGemma"),
         ("gte", "gte_kron_gc", "GTE-large-en-v1.5"),
         ("icae", "icae_genkron", "ICAE (Mistral-7B)")]
TESTS = ([(f, t, lbl) for f in FIELDS for t, lbl in
          (("collab", "collab AUC"), ("np", "next-paper AUC"), ("topic", "topic macro-F1"))]
         + [(ds, "name_disambig", "B$^3$-F1") for ds in S2AND])


def get(field, task, method):
    r = A[(A.field == field) & (A.task == task) & (A.method == method)]
    return (r.value.iloc[0], r.lo.iloc[0], r.hi.iloc[0]) if len(r) else (np.nan, np.nan, np.nan)


def _nz(v, dec=3):
    """Drop the leading zero on a quantity bounded by 1, matching Table 1."""
    s = f"{v:.{dec}f}"
    return s.replace("0.", ".", 1) if abs(v) < 1 else s


def fmt(v, lo, hi):
    return ("---" if not np.isfinite(v)
            else f"{_nz(v)}\\,{{\\tiny[{_nz(lo)},{_nz(hi)}]}}")


# ── tidy long CSV: raw, adapted and the delta for every (test, space) ────────────────────
rows = []
for field, task, _ in TESTS:
    for raw, adapted, label in PAIRS:
        v0 = get(field, task, raw)
        v1 = get(field, task, adapted)
        rows.append({"field": field, "task": task, "space": label, "raw": v0[0],
                     "raw_lo": v0[1], "raw_hi": v0[2], "adapted": v1[0], "adapted_lo": v1[1],
                     "adapted_hi": v1[2], "delta": v1[0] - v0[0]})
out = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
out.to_csv(OUT_CSV, index=False)

print("\n=== mean delta from g_theta, by space (over the tests with both numbers) ===")
print(out.dropna(subset=["delta"]).groupby("space").delta.agg(["mean", "min", "max", "count"])
      .to_string())
print("\n=== mean delta by task ===")
print(out.dropna(subset=["delta"]).groupby("task").delta.agg(["mean", "count"]).to_string())


def block(fh, tests, header):
    fh.write("\\begin{tabular}{l" + "r" * len(tests) + "}\n\\toprule\n")
    fh.write("space" + "".join(f" & {h}" for _, _, h in tests) + " \\\\\n")
    fh.write(header + " \\\\\n\\midrule\n")
    for raw, adapted, label in PAIRS:
        fh.write(label + " & " + " & ".join(fmt(*get(f, t, raw)) for f, t, _ in tests) + " \\\\\n")
        fh.write("\\quad $+g_\\theta$ & "
                 + " & ".join(fmt(*get(f, t, adapted)) for f, t, _ in tests) + " \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")


field_tests = [t for t in TESTS if t[1] != "name_disambig"]
s2and_tests = [t for t in TESTS if t[1] == "name_disambig"]
with open(OUT_TEX, "w") as fh:
    fh.write("% auto-generated by workflow/scripts/groupc/gcb_symmetric_table.py (#145) -- do not edit\n")
    block(fh, field_tests, "".join(f" & {f}" for f, _, _ in field_tests))
    fh.write("\n\\vspace{1em}\n\n")
    block(fh, s2and_tests, "".join(f" & {f}" for f, _, _ in s2and_tests))
# ── values-only block for the appendix: 14 columns fit once the intervals come out ──────
SHORT = {"aps": "Phys", "economics": "Econ", "psychology": "Psy", "zbmath": "zbMATH",
         "qian": "qian", "arnetminer": "arnet", "pubmed": "pubmed", "kisti": "kisti"}
TASK = {"collab": "col", "np": "np", "topic": "top", "name_disambig": "B$^3$"}
with open(OUT_VALUES, "w") as fh:
    fh.write("% auto-generated by workflow/scripts/groupc/gcb_symmetric_table.py (#145) -- do not edit\n")
    fh.write("\\begin{tabular}{l" + "r" * len(TESTS) + "}\n\\toprule\n")
    fh.write("space" + "".join(f" & {SHORT[f]}" for f, _, _ in TESTS) + " \\\\\n")
    fh.write("".join(f" & {TASK[t]}" for _, t, _ in TESTS) + " \\\\\n\\midrule\n")
    for raw, adapted, label in PAIRS:
        for col, tag in ((raw, label), (adapted, "\\quad $+g_\\theta$")):
            cells = []
            for f, t, _ in TESTS:
                v = get(f, t, col)[0]
                cells.append("---" if not np.isfinite(v) else _nz(v))
            fh.write(tag + " & " + " & ".join(cells) + " \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
print(f"\n[saved] {OUT_TEX} + {OUT_VALUES} + {OUT_CSV}", flush=True)
