"""Prep one S2AND sub-dataset: build the signature table + paper text + aligned specter vectors.
Usage: python prep.py zbmath
Outputs (data/s2and/proc<ds>/):
  sig_table.parquet  (signature_id, paper_id, block, cluster_id)
  paper_text.parquet (paper_id, text)
  specter.npz        (paper_ids[str], vecs[N,768])
"""
import os, sys, json
import numpy as np
import pandas as pd

DS = sys.argv[1] if len(sys.argv) > 1 else "zbmath"
RAW = f"data/{DS}"
OUT = f"proc/{DS}"
os.makedirs(f"data/s2and/{OUT}", exist_ok=True)
base = "data/s2and"

sig = json.load(open(f"{base}/{RAW}/{DS}_signatures.json"))
pap = json.load(open(f"{base}/{RAW}/{DS}_papers.json"))
clu = json.load(open(f"{base}/{RAW}/{DS}_clusters.json"))

# gold: signature_id -> cluster_id
sig2clu = {}
for cid, c in clu.items():
    for s in c["signature_ids"]:
        sig2clu[str(s)] = cid

rows = []
for sid, s in sig.items():
    sid = str(sid)
    rows.append(dict(signature_id=sid, paper_id=str(s["paper_id"]),
                     block=s["author_info"]["block"], cluster_id=sig2clu.get(sid)))
st = pd.DataFrame(rows).dropna(subset=["cluster_id"])
st.to_parquet(f"{base}/{OUT}/sig_table.parquet")

# paper text = title + abstract (abstract optional)
ptxt = []
need = set(st.paper_id)
for pid, p in pap.items():
    if str(pid) not in need:
        continue
    t = (p.get("title") or "").strip()
    a = (p.get("abstract") or "").strip()
    txt = (t + (". " + a if a else "")).strip()
    ptxt.append(dict(paper_id=str(pid), text=txt))
pt = pd.DataFrame(ptxt)
pt.to_parquet(f"{base}/{OUT}/paper_text.parquet")

# specter
mat, ids = (lambda sp: sp)(__import__("pickle").load(open(f"{base}/{RAW}/{DS}_specter.pickle", "rb")))
ids = np.asarray([str(x) for x in ids])
np.savez(f"{base}/{OUT}/specter.npz", paper_ids=ids, vecs=np.asarray(mat, np.float32))

print(f"[{DS}] sigs={len(st)} (of {len(sig)}) | papers_text={len(pt)} | specter={len(ids)} | blocks={st.block.nunique()} | clusters={st.cluster_id.nunique()}")
