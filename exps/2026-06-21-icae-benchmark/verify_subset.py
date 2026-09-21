"""[CPU] Prove the slice is lossless: every row in each bench npz is byte-identical to the matching
paper_id row in the full source npz. If so, any paper_id-keyed eval (collab/np/topic, S2AND) yields
identical numbers on bench vs full -- the only difference is which rows are present, and the eval only
ever touches rows in the subset.

Usage: python verify_subset.py field economics   |   python verify_subset.py s2and zbmath
"""
import sys, glob, os
import numpy as np

sys.path.insert(0, "workflow/scripts")
from bench_data import load

mode, name = sys.argv[1], sys.argv[2]
if mode == "field":
    _, _, src = load(name); bench = f"data/bench/{name}/embeddings"
else:
    src = f"exps/2026-06-09-s2and/proc/{name}"; bench = f"data/bench/s2and/{name}"

allok = True
for bf in sorted(glob.glob(f"{bench}/*.npz")):
    fn = os.path.basename(bf)
    sf = f"{src}/{fn}"
    if not os.path.exists(sf):
        print(f"{fn:44} NO SOURCE"); allok = False; continue
    zb = np.load(bf, allow_pickle=True); zs = np.load(sf, allow_pickle=True)
    key = "embeddings" if "embeddings" in zb.files else "vecs"
    srow = {str(p): i for i, p in enumerate(zs["paper_ids"])}
    mapped = np.array([srow[str(p)] for p in zb["paper_ids"]])
    ok = bool(np.array_equal(zb[key], zs[key][mapped]))
    print(f"{fn:44} bench {len(zb['paper_ids']):>8,} rows  byte-identical={ok}")
    allok &= ok
print("ALL ROWS BYTE-IDENTICAL -> eval results will match exactly" if allok else "MISMATCH!!")
