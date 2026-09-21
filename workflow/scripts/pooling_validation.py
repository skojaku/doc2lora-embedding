"""Pooling validation for the pre-head (norm_lora_emb) embedding.

Measures how well each pooling strategy preserves pairwise document similarity
structure (Spearman ρ vs. the full [n_layers, n_modules, r, latent] tensor).

Runs on Gemma, Qwen, and Mistral doc2lora checkpoints sequentially on GPU 0.

Usage (inside Docker):
    CUDA_VISIBLE_DEVICES=0 python pooling_validation.py \
        --paper-text /workspace/data/aps/paper_text.parquet \
        --output results/pooling_spearman.csv \
        --n-docs 500 --seed 42

Snakemake: reads input/output/params from snakemake object.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

CHECKPOINTS = {
    "gemma": "/opt/doc-to-lora/trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin",
    "qwen": "/opt/doc-to-lora/trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin",
    "mistral": "/opt/doc-to-lora/trained_d2l/mistral_7b_d2l/checkpoint-20000/pytorch_model.bin",
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path):
    from ctx_to_lora.model_loading import get_tokenizer
    from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

    state_dict = torch.load(checkpoint_path, weights_only=False, map_location="cuda")
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict, train=False, use_sequence_packing=False
    )
    model.eval()
    ctx_tokenizer = get_tokenizer(model.ctx_encoder.base_model.name_or_path)
    return model, ctx_tokenizer


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "libs" / "doc2lora"))

from doc2lora.embed import extract_norm_lora_emb_batch


def extract_embeddings(model, ctx_tokenizer, texts, batch_size=4):
    """Return list of [n_layers, n_modules, r, latent_size] tensors (CPU fp32)."""
    embs = extract_norm_lora_emb_batch(
        model, ctx_tokenizer, texts, batch_size=batch_size
    )
    return [e.cpu().float() for e in embs]


# ---------------------------------------------------------------------------
# Pooling strategies
# ---------------------------------------------------------------------------

def build_representations(embs):
    """
    Input: list of tensors [n_layers, n_modules, r, latent_size].
    Squeeze the singleton n_modules axis -> [n_layers, r, latent_size].

    Returns dict: strategy_name -> np.ndarray [n_docs, dim]
    """
    squeezed = []
    for e in embs:
        # e: [n_layers, n_modules, r, latent_size]
        assert e.shape[1] == 1, f"Expected singleton module axis, got shape {e.shape}"
        squeezed.append(e.squeeze(1))  # [n_layers, r, latent_size]

    n = len(squeezed)
    n_layers, r, d = squeezed[0].shape

    reps = {}

    # Full reference: [n_layers * r * latent_size]
    reps["full"] = np.stack([s.numpy().flatten() for s in squeezed])

    # Mean over rank (preserve layers): [n_layers, latent_size]
    reps["mean_over_rank"] = np.stack(
        [s.mean(dim=1).numpy().flatten() for s in squeezed]
    )

    # Mean over layers (preserve rank): [r, latent_size]
    reps["mean_over_layers"] = np.stack(
        [s.mean(dim=0).numpy().flatten() for s in squeezed]
    )

    # Mean over everything: [latent_size]
    reps["mean_over_all"] = np.stack(
        [s.mean(dim=(0, 1)).numpy().flatten() for s in squeezed]
    )

    # Single rank slice rank=0 (preserve layers): [n_layers, latent_size]
    reps["single_rank_slice"] = np.stack(
        [s[:, 0, :].numpy().flatten() for s in squeezed]
    )

    dims = {k: v.shape[1] for k, v in reps.items()}
    return reps, dims


# ---------------------------------------------------------------------------
# Pairwise cosine similarity
# ---------------------------------------------------------------------------

def pairwise_cosine(mat):
    """mat: [n, d] float32. Returns upper-triangle similarities as 1D array."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    normed = mat / np.maximum(norms, 1e-8)
    sim = normed @ normed.T
    idx = np.triu_indices(len(mat), k=1)
    return sim[idx]


# ---------------------------------------------------------------------------
# Per-model analysis
# ---------------------------------------------------------------------------

def run_model(model_name, checkpoint_path, texts, seed):
    print(f"\n=== {model_name} ===")
    print(f"  Loading checkpoint: {checkpoint_path}")
    model, ctx_tokenizer = load_model(checkpoint_path)

    print(f"  Extracting embeddings for {len(texts)} documents...")
    embs = extract_embeddings(model, ctx_tokenizer, texts)

    # Free GPU memory before computing correlations
    del model
    torch.cuda.empty_cache()

    print("  Building pooled representations...")
    reps, dims = build_representations(embs)

    ref_sim = pairwise_cosine(reps["full"].astype(np.float32))

    rows = []
    strategies = [
        ("full",             "Full [n_layers × r × latent]"),
        ("mean_over_rank",   "Mean over rank (preserve layers)"),
        ("mean_over_layers", "Mean over layers (preserve rank)"),
        ("mean_over_all",    "Mean over everything"),
        ("single_rank_slice","Single rank slice r=0 (preserve layers)"),
    ]

    for key, label in strategies:
        mat = reps[key].astype(np.float32)
        sim = pairwise_cosine(mat)
        if key == "full":
            rho = 1.0
        else:
            rho, _ = spearmanr(ref_sim, sim)
        dim = dims[key]
        print(f"  {label:<45s}  dim={dim:>8,}  ρ={rho:.4f}")
        rows.append({
            "model": model_name,
            "strategy": key,
            "label": label,
            "dim": dim,
            "spearman_rho": round(float(rho), 6),
            "n_docs": len(texts),
            "n_pairs": len(ref_sim),
            "seed": seed,
        })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(paper_text_path, output_path, n_docs, seed):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print(f"Loading APS corpus from {paper_text_path}")
    df = pd.read_parquet(paper_text_path)

    sample = df.sample(n=min(n_docs, len(df)), random_state=seed)
    texts = sample["text"].tolist()
    print(f"Sampled {len(texts)} documents (seed={seed})")

    all_rows = []
    for model_name, ckpt in CHECKPOINTS.items():
        rows = run_model(model_name, ckpt, texts, seed)
        all_rows.extend(rows)

    results = pd.DataFrame(all_rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    # Snakemake dual-mode
    try:
        paper_text_path = snakemake.input.paper_text
        output_path = snakemake.output.csv
        n_docs = snakemake.params.get("n_docs", 500)
        seed = snakemake.params.get("seed", 42)
    except NameError:
        parser = argparse.ArgumentParser()
        parser.add_argument("--paper-text", default="data/aps/paper_text.parquet")
        parser.add_argument("--output", default="results/pooling_spearman.csv")
        parser.add_argument("--n-docs", type=int, default=500)
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args()
        paper_text_path = args.paper_text
        output_path = args.output
        n_docs = args.n_docs
        seed = args.seed

    main(paper_text_path, output_path, n_docs, seed)
