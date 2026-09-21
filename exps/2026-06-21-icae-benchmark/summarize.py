"""Collect the ICAE-run result CSVs into one Markdown report: ICAE (raw) and icae_genkron (invertible
per-token adapter) alongside qwen genes/genkron and the text baselines, across field + S2AND tasks."""
import os
import pandas as pd

KR = "exps/2026-06-09-kron-adapter"
S2 = "exps/2026-06-09-s2and"
FIELDS = ["economics", "psychology", "aps"]
S2ANDS = ["zbmath", "qian", "arnetminer", "pubmed", "kisti"]

print("# ICAE benchmark results (vs doc2lora qwen genes + baselines)\n")
print("ICAE = raw mean-pooled 128 memory slots; `icae_genkron` = invertible per-token adapter "
      "(common rotation+scale + per-token scaling), trained on the same OpenAlex citation triplets "
      "as doc2lora's general adapter.\n")

print("## Field tasks (collab AUC / next-paper AUC / topic F1)\n")
for f in FIELDS:
    p = f"{KR}/results_{f}_qwen_icae.csv"
    if not os.path.exists(p):
        print(f"### {f}: MISSING ({p})\n"); continue
    df = pd.read_csv(p)
    print(f"### {f}\n")
    print(df.to_markdown(index=False))
    print()

print("## S2AND author disambiguation (B³ F1)\n")
rows = []
for ds in S2ANDS:
    p = f"{S2}/results_{ds}_qwen_icae.csv"
    if not os.path.exists(p):
        print(f"- {ds}: MISSING ({p})"); continue
    df = pd.read_csv(p).set_index("model")["B3_F1"]
    rows.append(df.rename(ds))
if rows:
    tab = pd.concat(rows, axis=1)
    print(tab.to_markdown())
