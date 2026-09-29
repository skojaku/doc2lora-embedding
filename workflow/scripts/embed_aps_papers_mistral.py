"""Generate doc2lora norm_lora_emb embeddings for APS papers using Mistral-7B.

Extracts norm_lora_emb — the L2-normalised vectors just before the LoRA head —
shape [32, 1, 8, 512] → flattens to 131,072-dim fp16.

Hook target: model.hypernet.head (input[0] = norm_lora_emb).

Processes papers in shards for checkpointing/resumability.
Supports multi-GPU parallelism: each GPU loads its own model and processes
assigned shards round-robin.

Usage (standalone):
    python embed_aps_papers_mistral.py \
        --paper-text data/aps/paper_text.parquet \
        --output data/aps/embeddings/mistral_norm_lora_emb.npz \
        --checkpoint-path /opt/doc-to-lora/trained_d2l/mistral_7b_d2l/checkpoint-20000/pytorch_model.bin \
        --shard-dir data/aps/embeddings/mistral_shards_norm_lora_emb \
        --shard-size 10000 \
        --gpu-ids 0,1,2,3

Snakemake: reads input/output/params from snakemake object.
"""

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

CHECKPOINT_PATH = "/opt/doc-to-lora/trained_d2l/mistral_7b_d2l/checkpoint-20000/pytorch_model.bin"


def load_model(checkpoint_path):
    """Load the doc2lora model (embed-only) and context tokenizer.

    Uses mode="embed", which frees the generator (base_model) after loading —
    embedding only needs the context encoder + hypernet. Cuts the steady-state
    footprint to ~6 GB (vs ~28 GB full), leaving headroom for larger batches.
    Peak VRAM during load is unchanged (full model is built, then half freed).
    """
    from doc2lora import load_model as _load_model

    model, _tokenizer, ctx_tokenizer = _load_model(
        checkpoint_path, device="cuda", mode="embed"
    )
    return model, ctx_tokenizer


def extract_norm_lora_emb(model, ctx_tokenizer, text, max_length=2048):
    """Extract norm_lora_emb for a single document.

    Hooks on model.hypernet.head and captures input[0], which is the
    L2-normalised embedding just before the LoRA head.

    Returns shape [n_layers, n_modules, r, latent_size].
    In practice: [32, 1, 8, 512] for the Mistral-7B doc2lora checkpoint.
    """
    chat = [{"role": "user", "content": text.strip()}]
    ctx_ids = ctx_tokenizer.apply_chat_template(
        chat,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)

    if ctx_ids.shape[1] > max_length:
        ctx_ids = ctx_ids[:, :max_length]

    captured = {}

    def hook_fn(module, input, output):
        captured["emb"] = input[0].detach()

    hook = model.hypernet.head.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            ctx_attn_mask = torch.ones_like(ctx_ids)
            model.generate_weights(ctx_ids, ctx_attn_mask)
    finally:
        hook.remove()

    return captured["emb"].squeeze(0)  # [n_layers, n_modules, r, latent_size]


def mean_pool_and_flatten(emb):
    """Mean-pool over r and n_modules, flatten over layers.

    Input: [n_layers, n_modules, r, latent_size] e.g. [32, 1, 8, 512]
    Output: [n_layers * latent_size] e.g. [16384]
    """
    pooled = emb.mean(dim=(1, 2))  # [n_layers, latent_size]
    return pooled.flatten().cpu().to(torch.float16).numpy()


def process_shard(model, ctx_tokenizer, texts, paper_ids, batch_size=2, gpu_label=""):
    """Process a shard of papers and return embeddings."""
    from doc2lora.embed import extract_norm_lora_emb_batch

    all_embs = []
    for start in tqdm(range(0, len(texts), batch_size), desc=f"{gpu_label}Embedding", leave=False):
        batch = texts[start : start + batch_size]
        embs = extract_norm_lora_emb_batch(model, ctx_tokenizer, batch, batch_size=batch_size)
        all_embs.extend([mean_pool_and_flatten(e) for e in embs])

    return np.stack(all_embs), np.array(paper_ids)


def gpu_worker(gpu_id, shard_indices, checkpoint_path, paper_text_path, shard_dir_str, shard_size):
    """Worker: load model on one GPU, process assigned shards."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    df = pd.read_parquet(paper_text_path)
    df = df.sort_values("aps_paper_id").reset_index(drop=True)
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
            print(f"{gpu_label}Shard {shard_idx}/{n_shards} already exists, skipping.")
            continue

        start = shard_idx * shard_size
        end = min(start + shard_size, n_papers)
        shard_df = df.iloc[start:end]

        print(f"{gpu_label}Processing shard {shard_idx + 1}/{n_shards} ({start}–{end})...")
        embeddings, paper_ids = process_shard(
            model,
            ctx_tokenizer,
            shard_df["text"].tolist(),
            shard_df["aps_paper_id"].tolist(),
            gpu_label=gpu_label,
        )

        np.savez_compressed(shard_path, embeddings=embeddings, paper_ids=paper_ids)
        print(f"{gpu_label}Saved shard {shard_idx + 1}/{n_shards}.")

    print(f"{gpu_label}All assigned shards complete.")


def main(
    paper_text_path,
    output_path,
    checkpoint_path,
    shard_dir,
    shard_size,
    gpu_ids=None,
):
    df = pd.read_parquet(paper_text_path)
    df = df.sort_values("aps_paper_id").reset_index(drop=True)
    n_papers = len(df)
    n_shards = (n_papers + shard_size - 1) // shard_size
    print(f"Total papers: {n_papers:,} → {n_shards} shards")

    shard_dir_path = Path(shard_dir)
    shard_dir_path.mkdir(parents=True, exist_ok=True)

    gpu_list = [g.strip() for g in gpu_ids.split(",")] if gpu_ids else ["0"]
    n_gpus = len(gpu_list)

    shard_assignments = [[] for _ in range(n_gpus)]
    for i in range(n_shards):
        shard_assignments[i % n_gpus].append(i)

    print(f"Using {n_gpus} GPU(s): {gpu_list}")
    for rank, gpu_id in enumerate(gpu_list):
        print(f"  GPU {gpu_id}: {len(shard_assignments[rank])} shards")

    if n_gpus > 1:
        mp.set_start_method("spawn", force=True)
        processes = []
        for rank in range(n_gpus):
            p = mp.Process(
                target=gpu_worker,
                args=(
                    gpu_list[rank],
                    shard_assignments[rank],
                    checkpoint_path,
                    paper_text_path,
                    str(shard_dir_path),
                    shard_size,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Worker exited with code {p.exitcode}")
    else:
        gpu_worker(
            gpu_list[0],
            list(range(n_shards)),
            checkpoint_path,
            paper_text_path,
            str(shard_dir_path),
            shard_size,
        )

    print("Concatenating shards...")
    all_embeddings = []
    all_paper_ids = []
    for shard_idx in range(n_shards):
        data = np.load(shard_dir_path / f"shard_{shard_idx:04d}.npz")
        all_embeddings.append(data["embeddings"])
        all_paper_ids.append(data["paper_ids"])

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_paper_ids = np.concatenate(all_paper_ids, axis=0)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=all_embeddings,
        paper_ids=all_paper_ids,
    )
    print(
        f"Saved {all_embeddings.shape[0]:,} embeddings "
        f"(shape {all_embeddings.shape}) to {output_path}"
    )


if __name__ == "__main__":
    try:
        main(
            paper_text_path=snakemake.input.paper_text,
            output_path=snakemake.output.embeddings,
            checkpoint_path=snakemake.params.checkpoint_path,
            shard_dir=snakemake.params.shard_dir,
            shard_size=snakemake.params.shard_size,
            gpu_ids=snakemake.params.get("gpu_ids", None),
        )
    except NameError:
        parser = argparse.ArgumentParser()
        parser.add_argument("--paper-text", required=True)
        parser.add_argument("--output", required=True)
        parser.add_argument("--checkpoint-path", default=CHECKPOINT_PATH)
        parser.add_argument("--shard-dir", default="data/aps/embeddings/mistral_shards_norm_lora_emb")
        parser.add_argument("--shard-size", type=int, default=10000)
        parser.add_argument("--gpu-ids", default=None)
        args = parser.parse_args()
        main(
            args.paper_text,
            args.output,
            args.checkpoint_path,
            args.shard_dir,
            args.shard_size,
            args.gpu_ids,
        )
