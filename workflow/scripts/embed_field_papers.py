"""[GPU] Generic doc2lora norm_lora_emb embedder for any field corpus & any model.

Generalises embed_aps_papers_{qwen,gemma,mistral}.py: pooling (mean over modules &
rank, flatten layers) is identical across checkpoints — only the checkpoint path and
the id column differ. Multi-GPU shard parallelism + resumable shards preserved.

Usage:
    python embed_field_papers.py \
        --paper-text data/fields/economics/paper_text.parquet \
        --output data/fields/economics/embeddings/qwen_norm_lora_emb.npz \
        --shard-dir data/fields/economics/embeddings/qwen_shards \
        --checkpoint-path data/agent_assets/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin \
        --id-col paper_id --shard-size 10000 --batch-size 4 --gpu-ids 0,1,2,3
"""
import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def load_model(checkpoint_path):
    """Embed-only load (mode='embed' frees the generator half) — cuts steady-state
    VRAM to ~6 GB so mistral-7B fits comfortably and larger batches are safe."""
    from doc2lora import load_model as _load_model

    model, _tokenizer, ctx_tokenizer = _load_model(checkpoint_path, device="cuda", mode="embed")
    return model, ctx_tokenizer


def mean_pool_and_flatten(emb):
    pooled = emb.mean(dim=(1, 2))  # [n_layers, latent]
    return pooled.flatten().cpu().to(torch.float16).numpy()


def process_shard(model, ctx_tokenizer, texts, paper_ids, max_length=768,
                  max_batch_tokens=16384, gpu_label=""):
    """Dynamic length-sorted batching: pass the whole shard and let
    extract_norm_lora_emb_batch pack batches up to a token budget (returns
    embeddings in the original input order, so paper_ids stay aligned)."""
    from doc2lora.embed import extract_norm_lora_emb_batch

    embs = extract_norm_lora_emb_batch(
        model, ctx_tokenizer, texts, max_length=max_length,
        max_batch_tokens=max_batch_tokens, sort_by_length=True)
    all_embs = [mean_pool_and_flatten(e) for e in embs]
    return np.stack(all_embs), np.array(paper_ids)


def gpu_worker(gpu_id, shard_indices, checkpoint_path, paper_text_path, shard_dir_str,
               shard_size, id_col, max_length, max_batch_tokens):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    df = pd.read_parquet(paper_text_path).sort_values(id_col).reset_index(drop=True)
    n_papers = len(df)
    n_shards = (n_papers + shard_size - 1) // shard_size
    shard_dir = Path(shard_dir_str)
    gpu_label = f"[GPU {gpu_id}] "

    print(f"{gpu_label}Loading model...")
    model, ctx_tokenizer = load_model(checkpoint_path)
    print(f"{gpu_label}Model loaded. Assigned {len(shard_indices)} shards.")

    for shard_idx in shard_indices:
        shard_path = shard_dir / f"shard_{shard_idx:04d}.npz"
        if shard_path.exists():
            print(f"{gpu_label}Shard {shard_idx}/{n_shards} exists, skipping.")
            continue
        start = shard_idx * shard_size
        end = min(start + shard_size, n_papers)
        shard_df = df.iloc[start:end]
        print(f"{gpu_label}Processing shard {shard_idx + 1}/{n_shards} ({start}-{end})...")
        embeddings, paper_ids = process_shard(
            model, ctx_tokenizer, shard_df["text"].tolist(),
            shard_df[id_col].tolist(), max_length=max_length,
            max_batch_tokens=max_batch_tokens, gpu_label=gpu_label)
        np.savez_compressed(shard_path, embeddings=embeddings, paper_ids=paper_ids)
        print(f"{gpu_label}Saved shard {shard_idx + 1}/{n_shards}.")
    print(f"{gpu_label}All assigned shards complete.")


def main(paper_text_path, output_path, checkpoint_path, shard_dir, shard_size,
         id_col="paper_id", max_length=768, max_batch_tokens=16384, gpu_ids=None):
    df = pd.read_parquet(paper_text_path).sort_values(id_col).reset_index(drop=True)
    n_papers = len(df)
    n_shards = (n_papers + shard_size - 1) // shard_size
    print(f"Total papers: {n_papers:,} -> {n_shards} shards (id_col={id_col})")

    shard_dir_path = Path(shard_dir)
    shard_dir_path.mkdir(parents=True, exist_ok=True)

    gpu_list = [g.strip() for g in gpu_ids.split(",")] if gpu_ids else ["0"]
    n_gpus = len(gpu_list)
    shard_assignments = [[] for _ in range(n_gpus)]
    for i in range(n_shards):
        shard_assignments[i % n_gpus].append(i)
    print(f"Using {n_gpus} GPU(s): {gpu_list}")

    if n_gpus > 1:
        mp.set_start_method("spawn", force=True)
        procs = []
        for rank in range(n_gpus):
            p = mp.Process(target=gpu_worker, args=(
                gpu_list[rank], shard_assignments[rank], checkpoint_path, paper_text_path,
                str(shard_dir_path), shard_size, id_col, max_length, max_batch_tokens))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Worker exited with code {p.exitcode}")
    else:
        gpu_worker(gpu_list[0], list(range(n_shards)), checkpoint_path, paper_text_path,
                   str(shard_dir_path), shard_size, id_col, max_length, max_batch_tokens)

    print("Concatenating shards...")
    all_embeddings, all_paper_ids = [], []
    for shard_idx in range(n_shards):
        data = np.load(shard_dir_path / f"shard_{shard_idx:04d}.npz")
        all_embeddings.append(data["embeddings"])
        all_paper_ids.append(data["paper_ids"])
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_paper_ids = np.concatenate(all_paper_ids, axis=0)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, embeddings=all_embeddings, paper_ids=all_paper_ids)
    print(f"Saved {all_embeddings.shape[0]:,} embeddings (shape {all_embeddings.shape}) to {output_path}")


if __name__ == "__main__":
    if "snakemake" in sys.modules:
        sm = snakemake  # noqa: F821
        main(sm.input["paper_text"], sm.output["embeddings"],
             sm.params["checkpoint_path"], sm.params["shard_dir"],
             int(sm.params["shard_size"]), id_col=sm.params.get("id_col", "paper_id"),
             max_length=int(sm.params.get("max_length", 768)),
             max_batch_tokens=int(sm.params.get("max_batch_tokens", 16384)),
             gpu_ids=sm.params["gpu_ids"])
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("--paper-text", required=True)
        ap.add_argument("--output", required=True)
        ap.add_argument("--checkpoint-path", required=True)
        ap.add_argument("--shard-dir", required=True)
        ap.add_argument("--id-col", default="paper_id")
        ap.add_argument("--shard-size", type=int, default=10000)
        ap.add_argument("--max-length", type=int, default=768)
        ap.add_argument("--max-batch-tokens", type=int, default=16384)
        ap.add_argument("--gpu-ids", default=None)
        a = ap.parse_args()
        main(a.paper_text, a.output, a.checkpoint_path, a.shard_dir, a.shard_size,
             a.id_col, a.max_length, a.max_batch_tokens, a.gpu_ids)
