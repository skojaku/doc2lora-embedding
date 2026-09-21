"""Start from a PACS cluster mean, REDUCE its magnitude step by step, decode along
the way (norm_lora_emb head path, which IS magnitude-sensitive). Does the response
fall into a broad/general concept as the gene shrinks toward zero?

Uses the new doc2lora.cluster_summary API.
Run: CUDA_VISIBLE_DEVICES=0 ~/miniforge3/envs/doc2lora/bin/python magnitude_trajectory.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]   # repository root, not a fixed ~/projects path
CA = ROOT / "exps/2026-05-28-concept-analogy-aps"
for _p in ("libs/legacy", "libs/doc2lora"):   # doc2lora_legacy lives in libs/legacy
    sys.path.insert(0, str(ROOT / _p))
from doc2lora_legacy import load_model, cluster_centroid, decode_cluster  # noqa: E402

CKPT = ROOT / "data/agent_assets/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"
DISK = ROOT / "data/aps/embeddings/qwen_norm_lora_emb.npz"
LABEL = ("In 3 to 6 words, name the single scientific subfield or research topic that all of "
         "these documents share. Reply with only the topic name, nothing else.")
SCALES = [2.0, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.0]
SAMPLE = 4000
rng = np.random.default_rng(0)
CLUSTERS = [("root", "root", None, "Physics"), ("3", "main", 3, "Condensed matter"),
            ("42.50", "sub", "42.50", "Quantum optics"), ("7", "main", 7, "Nuclear physics"),
            ("1", "main", 1, "Atomic physics")]


def main():
    dz = np.load(DISK)
    genes = dz["embeddings"]
    pid2row = {int(p): i for i, p in enumerate(dz["paper_ids"])}
    pg = pd.read_parquet(CA / "results/paper_groups.parquet").dropna(
        subset=["main_class_id", "division", "subdivision"]).copy()
    pg["paper_id"] = pg.paper_id.astype(int)
    pg["main_class_id"] = pg.main_class_id.astype(int)
    pg["division"] = pg.division.astype(str)
    pg["subdivision"] = pg.subdivision.astype(str)

    def rows_of(level, val):
        if level == "root":
            ids = pg.paper_id.values
        elif level == "main":
            ids = pg[pg.main_class_id == val].paper_id.values
        else:
            ids = pg[pg.subdivision == val].paper_id.values
        r = np.array([pid2row[int(p)] for p in ids if int(p) in pid2row])
        return r if len(r) <= SAMPLE else rng.choice(r, SAMPLE, replace=False)

    model, _ctx, gen_tok = load_model(str(CKPT))
    print("model loaded\n", flush=True)

    for key, level, val, name in CLUSTERS:
        c = cluster_centroid(genes[rows_of(level, val)], op="mean", n_layers=36)
        print(f"================ [{key}] {name} ================", flush=True)
        for s in SCALES:
            lab = decode_cluster(model, gen_tok, c, LABEL, scale=s, renorm=False, max_new_tokens=24)
            lab = " ".join(lab.replace("*", "").replace('"', "").split())[:80]
            print(f"   scale={s:<5} {lab!r}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
