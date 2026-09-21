"""[GPU] Full-scale SBERT (all-mpnet) embeddings for every APS paper with a valid PACS1 code and
a usable abstract — the textual baseline for the idea-compatibility analysis, on the SAME papers
as the doc2lora genes (no per-code subsampling). Sharded + resumable.

Dual-mode. Out: npz {paper_ids, vecs}.
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

PACS_RE = re.compile(r"^\d\d\.\d\d")


def main(paper_text, paper_table, out_npz, model_name, min_chars=300, shard_dir=None, shard_size=20000):
    pt = pd.read_csv(paper_table, usecols=["paper_id", "PACS1"], dtype={"paper_id": np.int64, "PACS1": str})
    pt = pt[pt["PACS1"].notna() & pt["PACS1"].str.match(PACS_RE, na=False)]
    tx = pd.read_parquet(paper_text, columns=["aps_paper_id", "abstract"]).rename(columns={"aps_paper_id": "paper_id"})
    tx = tx[tx["abstract"].str.len().fillna(0) >= min_chars]
    df = pt.merge(tx, on="paper_id", how="inner").drop_duplicates("paper_id").sort_values("paper_id").reset_index(drop=True)
    print(f"[sbert] {len(df)} papers (valid PACS1 + abstract>={min_chars})")

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(model_name)
    shard_dir = shard_dir or (os.path.dirname(out_npz) + "/sbert_shards")
    os.makedirs(shard_dir, exist_ok=True)
    n = len(df)
    nshard = (n + shard_size - 1) // shard_size
    for sidx in range(nshard):
        sp = os.path.join(shard_dir, f"shard_{sidx:04d}.npz")
        if os.path.exists(sp):
            continue
        s, e = sidx * shard_size, min((sidx + 1) * shard_size, n)
        chunk = df.iloc[s:e]
        V = enc.encode(chunk["abstract"].tolist(), normalize_embeddings=True, convert_to_numpy=True,
                       batch_size=256, show_progress_bar=True).astype(np.float32)
        np.savez(sp, paper_ids=chunk["paper_id"].to_numpy(), vecs=V)
        print(f"[sbert] shard {sidx+1}/{nshard} ({s}-{e})")

    ids, vecs = [], []
    for sidx in range(nshard):
        z = np.load(os.path.join(shard_dir, f"shard_{sidx:04d}.npz"))
        ids.append(z["paper_ids"]); vecs.append(z["vecs"])
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez(out_npz, paper_ids=np.concatenate(ids), vecs=np.concatenate(vecs))
    print(f"[sbert] -> {out_npz}  {np.concatenate(vecs).shape}")


if __name__ == "__main__":
    if "snakemake" in sys.modules:
        sm = snakemake  # noqa: F821
        main(sm.input["paper_text"], sm.params["paper_table"], sm.output["embeddings"],
             sm.params["model_name"], min_chars=int(sm.params["min_chars"]),
             shard_dir=sm.params["shard_dir"], shard_size=int(sm.params["shard_size"]))
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("--paper-text", required=True)
        ap.add_argument("--paper-table", default="/data/datasets/aps/preprocessed/paper_table.csv")
        ap.add_argument("--out-npz", required=True)
        ap.add_argument("--model-name", default="sentence-transformers/all-mpnet-base-v2")
        ap.add_argument("--min-chars", type=int, default=300)
        ap.add_argument("--shard-dir", default=None)
        ap.add_argument("--shard-size", type=int, default=20000)
        a = ap.parse_args()
        main(a.paper_text, a.paper_table, a.out_npz, a.model_name, a.min_chars, a.shard_dir, a.shard_size)
