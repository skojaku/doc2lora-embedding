"""Model loading utilities for doc2lora."""

import gc

import torch

_VALID_MODES = ("full", "embed", "generate")


def load_model(checkpoint_path, device="cuda", mode="full"):
    """Load the doc2lora model and both tokenizers.

    The doc2lora model holds two Mistral-sized copies: a context encoder
    (reads documents -> embeddings) and a generator (the base model the
    generated LoRA is applied to for text). They serve different phases of the
    pipeline and are never needed at the same time, so ``mode`` lets you keep
    only the half a given phase needs resident on the GPU.

    Args:
        checkpoint_path: path to the doc2lora checkpoint (.bin).
        device: target device.
        mode: which half to keep on the GPU after loading:
            - ``"full"``     : both halves (default; backward compatible).
            - ``"embed"``    : keep the context encoder + hypernet, free the
                               generator (~18 GB for Mistral-7B). Use for
                               computing embeddings (``extract_*``).
            - ``"generate"`` : keep the generator + hypernet, free the context
                               encoder (~6 GB). Use for generating text from
                               precomputed embeddings
                               (``internalize_from_norm_lora_emb`` ->
                               ``generate_text``).

    Note:
        Loading always builds the full model first, so peak VRAM during
        ``load_model`` is the same in every mode; ``mode`` reduces the
        *steady-state* footprint after loading returns.

    Returns:
        model: ModulatedPretrainedModel (on device, eval mode)
        tokenizer: tokenizer for the base LLM (for generation)
        ctx_tokenizer: tokenizer for the context encoder (for embedding)

        Both tokenizers are returned in every mode (they are cheap and hold no
        GPU memory); only the freed half's *model* is dropped.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")

    from ctx_to_lora.model_loading import get_tokenizer
    from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

    state_dict = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict, train=False, use_sequence_packing=False
    )
    model.eval()

    # Resolve tokenizers before freeing either half (they read .name_or_path).
    tokenizer = get_tokenizer(model.base_model.name_or_path)
    ctx_tokenizer = get_tokenizer(model.ctx_encoder.base_model.name_or_path)

    if mode == "embed":
        # Generator not needed; embedding only uses ctx_encoder + hypernet.
        model.base_model = None
        model.generated_loras = None
    elif mode == "generate":
        # Context encoder not needed; text is generated from precomputed
        # embeddings via internalize_from_norm_lora_emb (head + base_model only).
        model.ctx_encoder = None

    if mode != "full":
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return model, tokenizer, ctx_tokenizer
