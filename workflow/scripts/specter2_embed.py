"""[GPU] True SPECTER2 (allenai/specter2_base + proximity adapter) embeddings for a field corpus.

Runs in a SEPARATE env (transformers>=4.57 + adapters) because the `adapters` library
conflicts with the doc2lora env's pinned transformers 4.51.3. Output is drop-in compatible
with the Snakemake baseline path: data/fields/<field>/embeddings/baseline_specter2.npz
with keys {paper_ids:int64, vecs:float32 [N,768]} keyed by field-local paper_id.

Input format per SPECTER2: title + tokenizer.sep_token + abstract; embedding = CLS token.

Usage:
    python specter2_embed.py --paper-text data/fields/economics/paper_text.parquet \
        --output data/fields/economics/embeddings/baseline_specter2.npz --batch-size 64 --gpu 0

NOT WIRED INTO A RULE: standalone diagnostic, kept for provenance. The reported
numbers come from the chains in workflow/rules/; see REPRODUCE.md.
"""
import argparse

import numpy as np
import pandas as pd
import torch


def main(paper_text, output, batch_size=64, gpu=0, max_length=512):
    from transformers import AutoTokenizer
    from adapters import AutoAdapterModel

    dev = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    # the proximity adapter (the one trained for paper-paper similarity)
    name = model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
    model.set_active_adapters(name)  # ensure it's active in the forward pass
    print(f"[specter2] loaded adapter={name!r}  active={model.active_adapters}", flush=True)
    assert model.active_adapters is not None, "proximity adapter is NOT active!"
    model.to(dev).eval()

    df = pd.read_parquet(paper_text)
    # APS keys on `aps_paper_id`, the field corpora on `paper_id`.
    if "paper_id" not in df.columns and "aps_paper_id" in df.columns:
        df = df.rename(columns={"aps_paper_id": "paper_id"})
    titles = df["title"].fillna("").astype(str).tolist()
    absts = df["abstract"].fillna("").astype(str).tolist()
    texts = [t + tok.sep_token + a for t, a in zip(titles, absts)]

    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inp = tok(batch, padding=True, truncation=True, return_tensors="pt",
                      max_length=max_length, return_token_type_ids=False).to(dev)
            out = model(**inp)
            emb = out.last_hidden_state[:, 0, :]  # CLS pooling (SPECTER2 convention)
            vecs.append(emb.cpu().float().numpy())
            if (i // batch_size) % 50 == 0:
                print(f"  {i + len(batch):,}/{len(texts):,}", flush=True)

    V = np.concatenate(vecs).astype(np.float32)
    pids = df["paper_id"].to_numpy(np.int64)
    import os
    os.makedirs(os.path.dirname(output), exist_ok=True)
    np.savez(output, paper_ids=pids, vecs=V)
    print(f"[specter2] -> {output}  vecs{V.shape}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-text", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=512)
    a = ap.parse_args()
    main(a.paper_text, a.output, a.batch_size, a.gpu, a.max_length)
