"""[CPU] Slice existing embedding .npz files down to the eval-touched paper subset (NO recompute),
mirroring the ICAE subset trick for doc2lora and every baseline. Writes to a separate bench tree so
the full source datasets -- including the standalone full APS at data/aps used by another experiment --
are left untouched.

  field <f>   data/fields/<f>/embeddings/*.npz  (aps: data/aps/embeddings) -> data/bench/<f>/embeddings/
              keep only rows whose paper_id is in <f>_eval_ids.parquet (topic+np+collab union).
  s2and <ds>  data/s2and/proc<ds>/*  -> data/bench/s2and/<ds>/   (whole dataset = eval set,
              so this is effectively a copy; metadata sig_table/paper_text copied too).

Usage: python make_subset.py field economics   |   python make_subset.py s2and zbmath
"""
import os, sys, glob, shutil
import numpy as np
import pandas as pd

sys.path.insert(0, "workflow/scripts")
from bench_data import load

IDS = "data/icae"     # eval-id parquets
BENCH = "data/bench"
S2 = "data/s2and"


def slice_one(src, dst, want):
    """Slice a single npz (key 'embeddings' or 'vecs' + 'paper_ids') to rows whose pid is in `want`
    (a set of str). Returns kept-count, or None if the file isn't a sliceable embedding npz."""
    z = np.load(src, allow_pickle=True)
    if "paper_ids" not in z.files:
        return None
    key = "embeddings" if "embeddings" in z.files else ("vecs" if "vecs" in z.files else None)
    if key is None:
        return None
    pid = z["paper_ids"]
    keep = np.fromiter((i for i, p in enumerate(pid) if str(p) in want), dtype=np.int64)
    np.savez(dst, **{key: z[key][keep], "paper_ids": pid[keep]})
    return len(keep)


def do_field(field):
    _, _, emb_dir = load(field)
    want = set(str(int(p)) for p in pd.read_parquet(f"{IDS}/{field}_eval_ids.parquet").paper_id)
    out = f"{BENCH}/{field}/embeddings"
    os.makedirs(out, exist_ok=True)
    print(f"[{field}] {len(want):,} eval ids; slicing {emb_dir} -> {out}", flush=True)
    for src in sorted(glob.glob(f"{emb_dir}/*.npz")):
        n = slice_one(src, f"{out}/{os.path.basename(src)}", want)
        tag = f"{n:,} rows" if n is not None else "skip (not embedding npz)"
        print(f"   {os.path.basename(src):42} {tag}", flush=True)


def do_s2and(ds):
    src = f"{S2}/proc/{ds}"
    out = f"{BENCH}/s2and/{ds}"
    os.makedirs(out, exist_ok=True)
    z = np.load(f"{src}/specter.npz", allow_pickle=True)
    want = set(str(p) for p in z["paper_ids"])          # whole dataset = eval set
    print(f"[s2and:{ds}] {len(want):,} papers; copying {src} -> {out}", flush=True)
    for f in sorted(glob.glob(f"{src}/*.npz")):
        n = slice_one(f, f"{out}/{os.path.basename(f)}", want)
        print(f"   {os.path.basename(f):28} {f'{n:,} rows' if n is not None else 'skip'}", flush=True)
    for meta in ["sig_table.parquet", "paper_text.parquet"]:
        if os.path.exists(f"{src}/{meta}"):
            shutil.copy2(f"{src}/{meta}", f"{out}/{meta}")
            print(f"   copied {meta}", flush=True)


if __name__ == "__main__":
    mode, name = sys.argv[1], sys.argv[2]
    (do_field if mode == "field" else do_s2and)(name)
