"""Generate full [26, 8, 512] doc2lora embeddings for a subset of papers.

These full embeddings are needed for LLM decoding (internalize_from_latents).

Usage (standalone):
    python embed_papers_full.py \
        --paper-text data/aps/paper_text.parquet \
        --paper-ids data/aps/paper_ids_subset.txt \
        --output data/aps/embeddings/doc2lora_full_subset.npz \
        --checkpoint-path $DOC2LORA_CKPT

Snakemake: reads input/output/params from snakemake object.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def load_model(checkpoint_path):
    """Load the doc2lora model and tokenizer."""
    from ctx_to_lora.model_loading import get_tokenizer
    from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

    state_dict = torch.load(checkpoint_path, weights_only=False, map_location="cuda")
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict, train=False, use_sequence_packing=False
    )
    model.eval()

    ctx_tokenizer = get_tokenizer(model.ctx_encoder.base_model.name_or_path)
    return model, ctx_tokenizer


def extract_encoder_latents(model, ctx_tokenizer, text, max_length=2048):
    """Extract perceiver encoder latents for a single document.

    Returns encoder latents with shape [n_layers, n_latents, hidden_dim].
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
        captured["latents"] = output.detach()

    hook = model.hypernet.aggregator.perceiver.encoder.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            ctx_attn_mask = torch.ones_like(ctx_ids)
            model.generate_weights(ctx_ids, ctx_attn_mask)
    finally:
        hook.remove()

    return captured["latents"]  # [n_layers, n_latents, hidden_dim]


def main(
    paper_text_path,
    paper_ids_path,
    output_path,
    checkpoint_path,
    gpu_ids=None,
):
    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    # Load paper IDs to embed
    paper_ids = np.loadtxt(paper_ids_path, dtype=int)
    print(f"Papers to embed: {len(paper_ids)}")

    # Load text data and filter to requested papers
    df = pd.read_parquet(paper_text_path)
    df = df[df["aps_paper_id"].isin(paper_ids)].sort_values("aps_paper_id").reset_index(drop=True)
    print(f"Matched papers: {len(df)}")

    # Load model
    print("Loading model...")
    model, ctx_tokenizer = load_model(checkpoint_path)
    print("Model loaded.")

    # Extract full embeddings
    all_latents = []
    found_ids = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding"):
        latents = extract_encoder_latents(model, ctx_tokenizer, row["text"])
        all_latents.append(latents.cpu().to(torch.float16).numpy())
        found_ids.append(row["aps_paper_id"])
        model.reset()

    # Stack: [n_papers, n_layers, n_latents, hidden_dim]
    all_latents = np.stack(all_latents)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=all_latents,
        paper_ids=np.array(found_ids),
    )
    print(f"Saved {all_latents.shape[0]} full embeddings (shape {all_latents.shape}) to {output_path}")


if __name__ == "__main__":
    try:
        main(
            paper_text_path=snakemake.input.paper_text,
            paper_ids_path=snakemake.input.paper_ids,
            output_path=snakemake.output.embeddings,
            checkpoint_path=snakemake.params.checkpoint_path,
            gpu_ids=snakemake.params.get("gpu_ids", None),
        )
    except NameError:
        parser = argparse.ArgumentParser()
        parser.add_argument("--paper-text", required=True)
        parser.add_argument("--paper-ids", required=True)
        parser.add_argument("--output", required=True)
        parser.add_argument(
            "--checkpoint-path",
            default=os.environ.get("DOC2LORA_CKPT"),
        )
        parser.add_argument("--gpu-ids", default=None)
        args = parser.parse_args()
        main(args.paper_text, args.paper_ids, args.output, args.checkpoint_path, args.gpu_ids)
