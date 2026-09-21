"""Merge the per-shard full-rank mean npz files into one qwen_fullrank_means.npz."""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
shards = sorted(HERE.glob("qwen_fullrank_means_s*of*.npz"))
means, nodes, n_used, counts = [], [], [], []
for sp in shards:
    z = np.load(sp, allow_pickle=True)
    means.append(z["means"])
    nodes.append(z["nodes"].astype(str))
    n_used.append(z["n_used"])
    counts.append(z["counts"])
    print(f"{sp.name}: {len(z['nodes'])} nodes")

means = np.concatenate(means)
nodes = np.concatenate(nodes)
n_used = np.concatenate(n_used)
counts = np.concatenate(counts)
out = HERE / "qwen_fullrank_means.npz"
np.savez(out, means=means.astype(np.float32), nodes=nodes,
         n_used=n_used, counts=counts)
print(f"wrote {out.name}: means {means.shape} over {len(nodes)} nodes")
for k, nu, c, m in zip(nodes, n_used, counts, means):
    print(f"  {k:>6} n_used={nu:>5}/{c:<7} ||mean||={np.linalg.norm(m):.3f}")
