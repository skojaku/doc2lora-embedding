"""Compute the FULL-RANK mean qwen norm_lora_emb per PACS node.

The stored qwen_norm_lora_emb_raw.npz is POOLED over the 8 rank slots, so the
full-rank [L=36, r=8, latent=512] structure is not recoverable from it. Here we
re-extract the full-rank norm_lora_emb per paper and average over node members.

Per the project finding (sampled mean ~= true full-corpus mean) we sample up to
CAP members per node; set CAP huge to use all members. Saves incrementally so the
job is resumable (re-run skips nodes already in the npz).

Output: qwen_fullrank_means.npz
  means : [n_nodes, 36, 8, 512] float32   (full-rank mean, rank slots preserved)
  nodes : [n_nodes] str
  n_used: [n_nodes] int
  counts: [n_nodes] int   (total members available)

Run (GPU): CUDA_VISIBLE_DEVICES=0 python compute_qwen_fullrank_means.py [CAP]
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]   # repository root, not a fixed ~/projects path
CA = ROOT / "exps/2026-05-28-concept-analogy-aps"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "libs/doc2lora"))
from doc2lora_legacy import load_model  # noqa: E402
from doc2lora_legacy.embed import extract_norm_lora_emb_batch  # noqa: E402

CKPT = ROOT / "data/agent_assets/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"
CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
# token-budget batching: cap total tokens/batch so memory is bounded regardless
# of doc length (fixed batch_size OOMs on batches of long docs).
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 6144
CHUNK = 512  # members fed to the batched extractor per accumulation step
# optional shard arg "idx/n" -> this process handles nodes[idx::n] into its own file
SHARD = sys.argv[2] if len(sys.argv) > 2 else None
if SHARD:
    _i, _n = (int(x) for x in SHARD.split("/"))
    OUT = HERE / f"qwen_fullrank_means_s{_i}of{_n}.npz"
else:
    _i, _n = 0, 1
    OUT = HERE / "qwen_fullrank_means.npz"
SEED = 0
L, R, LAT = 36, 8, 512

# (node_key, level, value) -- same nodes as the decoded tree, plus node 8.
SPEC = [
    {"fid": 3, "divs": [("75", ["75.10", "75.30"]), ("71", ["71.10", "71.20"])]},
    {"fid": 0, "divs": [("03", ["03.67", "03.65"]), ("05", ["05.45", "05.40"])]},
    {"fid": 6, "divs": [("11", ["11.15", "11.10"]), ("12", ["12.38", "12.60"])]},
    {"fid": 2, "divs": [("42", ["42.50", "42.65"]), ("47", ["47.27", "47.20"])]},
]


def node_list():
    nodes = [("root", "root", None)]
    for f in SPEC:
        nodes.append((str(f["fid"]), "main", f["fid"]))
        for code, subs in f["divs"]:
            nodes.append((code, "division", code))
            for s in subs:
                nodes.append((s, "subdivision", s))
    nodes.append(("8", "main", 8))
    return nodes


def main():
    pg = pd.read_parquet(CA / "results/paper_groups.parquet")
    txt = pd.read_parquet(ROOT / "data/aps/paper_text.parquet")
    idc = "aps_paper_id" if "aps_paper_id" in txt.columns else "paper_id"
    id2t = dict(zip(txt[idc].astype(int), txt["text"].astype(str)))
    rng = np.random.default_rng(SEED)

    def members(level, val):
        if level == "root":
            m = pg.paper_id.values
        elif level == "main":
            m = pg[pg.main_class_id == val].paper_id.values
        elif level == "division":
            m = pg[pg.division.astype(str) == val].paper_id.values
        else:
            m = pg[pg.subdivision.astype(str) == val].paper_id.values
        return np.array([int(p) for p in m if int(p) in id2t])

    # resume
    store = {}
    if OUT.exists():
        z = np.load(OUT, allow_pickle=True)
        for i, k in enumerate(z["nodes"]):
            store[str(k)] = (z["means"][i], int(z["n_used"][i]), int(z["counts"][i]))
        print(f"resume: {len(store)} nodes already done", flush=True)

    nodes = node_list()[_i::_n]   # this shard's nodes
    todo = [n for n in nodes if n[0] not in store]
    if not todo:
        print("all nodes already computed", flush=True)
        return

    model, gen_tok, ctx_tok = load_model(str(CKPT), mode="full")
    print("model loaded", flush=True)

    def save():
        keys = [k for k, _, _ in nodes if k in store]
        means = np.stack([store[k][0] for k in keys]).astype(np.float32)
        n_used = np.array([store[k][1] for k in keys])
        counts = np.array([store[k][2] for k in keys])
        np.savez(OUT, means=means, nodes=np.array(keys),
                 n_used=n_used, counts=counts)

    for key, level, val in todo:
        pids = members(level, val)
        total = len(pids)
        if total == 0:
            print(f"  {key}: no members", flush=True)
            continue
        sel = pids if total <= CAP else rng.choice(pids, size=CAP, replace=False)
        acc = torch.zeros(L, 1, R, LAT, dtype=torch.float64)
        for c0 in range(0, len(sel), CHUNK):
            chunk = sel[c0:c0 + CHUNK]
            embs = extract_norm_lora_emb_batch(
                model, ctx_tok, [id2t[int(p)] for p in chunk],
                max_batch_tokens=MAXTOK)
            acc += torch.stack([e.cpu() for e in embs]).double().sum(0)
            print(f"    {key}: {min(c0 + CHUNK, len(sel))}/{len(sel)}", flush=True)
        mean = (acc / len(sel)).squeeze(1).float().numpy()   # [36, 8, 512]
        store[key] = (mean, int(len(sel)), int(total))
        save()
        print(f"  [{key}] n_used={len(sel)}/{total} "
              f"||mean||={np.linalg.norm(mean):.3f}", flush=True)

    print(f"wrote {OUT.name} ({len(store)} nodes)", flush=True)


if __name__ == "__main__":
    main()
