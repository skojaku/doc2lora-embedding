"""Model loading utilities for doc2lora."""

import gc
import glob
import os

import torch

from doc2lora.device import resolve_device

_VALID_MODES = ("full", "embed", "generate")


def resolve_checkpoint(checkpoint_path=None):
    """Resolve the doc2lora checkpoint path with a single canonical rule.

    Resolution order (first hit wins):

    1. an explicit ``checkpoint_path`` argument,
    2. the ``$DOC2LORA_CKPT`` environment variable,
    3. a repo-relative fallback under ``data/agent_assets/*/checkpoint-*/pytorch_model.bin``.

    Args:
        checkpoint_path: explicit path, or ``None`` to auto-resolve.

    Returns:
        str: a path to an existing checkpoint ``.bin``.

    Raises:
        FileNotFoundError: if nothing resolves, listing every location tried.
    """
    if checkpoint_path:
        return checkpoint_path

    env = os.environ.get("DOC2LORA_CKPT")
    if env:
        return env

    matches = sorted(glob.glob("data/agent_assets/*/checkpoint-*/pytorch_model.bin"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        "No doc2lora checkpoint found. Pass checkpoint_path=..., set "
        "$DOC2LORA_CKPT, or place one under "
        "data/agent_assets/<model>/checkpoint-<step>/pytorch_model.bin."
    )


def _import_backbone():
    """Import the ``ctx_to_lora`` backbone with an actionable error message.

    ``ctx_to_lora`` (and, on CUDA, ``flash-attn``) are heavy optional runtime
    dependencies that are not on PyPI, so a bare ``ImportError`` here is
    confusing. Surface the fix instead of a raw stack trace.
    """
    try:
        from ctx_to_lora.model_loading import get_tokenizer
        from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
    except ImportError as e:  # pragma: no cover - exercised only without the backbone
        raise ImportError(
            "doc2lora needs the `ctx_to_lora` backbone (and, on CUDA, "
            "`flash-attn`), which are not on PyPI. Install the doc-to-lora repo "
            "(`pip install -e /path/to/doc-to-lora`) and, for GPU, build the "
            "flash-attn wheel — see the doc2lora README. Original error: "
            f"{e}"
        ) from e
    return get_tokenizer, ModulatedPretrainedModel


def load_model(checkpoint_path=None, device=None, mode="full"):
    """Load the doc2lora model and both tokenizers.

    The doc2lora model holds two full-model-sized copies: a context encoder
    (reads documents -> embeddings) and a generator (the base model the
    generated LoRA is applied to for text). The checkpoint decides the backbone
    (Gemma / Mistral / Qwen); this loader is checkpoint-agnostic. The two halves
    serve different phases of the pipeline and are never needed at the same
    time, so ``mode`` lets you keep only the half a given phase needs resident.

    Args:
        checkpoint_path: path to the doc2lora checkpoint (.bin), or ``None`` to
            resolve via :func:`resolve_checkpoint` (``$DOC2LORA_CKPT`` then a
            repo-relative ``data/agent_assets/`` fallback).
        device: target device. ``None`` auto-selects CUDA / MPS / CPU (see
            :func:`doc2lora.device.resolve_device`); pass ``"cpu"`` to force it.
        mode: which half to keep resident after loading:
            - ``"full"``     : both halves (default; backward compatible).
            - ``"embed"``    : keep the context encoder + hypernet, free the
                               generator. Use for computing embeddings
                               (``extract_*``).
            - ``"generate"`` : keep the generator + hypernet, free the context
                               encoder. Use for generating text from
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

    checkpoint_path = resolve_checkpoint(checkpoint_path)
    device = resolve_device(device)
    get_tokenizer, ModulatedPretrainedModel = _import_backbone()

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
