"""Embedding extraction from the doc2lora perceiver encoder."""

import numpy as np
import torch


def _safe_reset(model):
    """Undo LoRA forward-patching, tolerating an embed-only model.

    ``model.reset()`` walks ``base_model`` layers to restore their original
    forward functions. Embedding never patches ``base_model`` (it only hooks the
    hypernet head), so this is pure defensive cleanup. In ``mode="embed"`` the
    generator is freed (``base_model is None``), where the real reset would
    crash — so we skip it and just clear the cached-lora bookkeeping.
    """
    if getattr(model, "base_model", None) is not None:
        model.reset()
    else:
        model.generated_loras = None


def _tokenize_single(ctx_tokenizer, text, max_length):
    """Tokenize a single document with chat template. Returns 1D tensor of token ids."""
    chat = [{"role": "user", "content": text.strip()}]
    ids = ctx_tokenizer.apply_chat_template(
        chat,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        add_special_tokens=False,
    ).squeeze(0)  # [seq_len]
    if ids.shape[0] > max_length:
        ids = ids[:max_length]
    return ids


def _pad_and_stack(token_lists, pad_value=0, device="cpu"):
    """Pad variable-length token id tensors and stack into a batch.

    Returns:
        input_ids: [bs, max_len]
        attn_mask: [bs, max_len]
    """
    max_len = max(t.shape[0] for t in token_lists)
    batch_ids = []
    batch_mask = []
    for t in token_lists:
        pad_len = max_len - t.shape[0]
        padded = torch.nn.functional.pad(t, (0, pad_len), value=pad_value)
        mask = torch.cat([torch.ones(t.shape[0], dtype=torch.long),
                          torch.zeros(pad_len, dtype=torch.long)])
        batch_ids.append(padded)
        batch_mask.append(mask)
    return torch.stack(batch_ids).to(device), torch.stack(batch_mask).to(device)


def _make_batches(lengths, batch_size=8, max_batch_tokens=None):
    """Group document indices into batches over a length-sorted order.

    Sorting by token length keeps each batch length-homogeneous, so short
    documents are no longer padded up to a long neighbour (padding is per-batch,
    to the batch's longest doc). This cuts wasted compute and peak memory.

    Two batching policies:
      - ``max_batch_tokens`` set: token-budget / "dynamic" batching. Pack a
        variable number of docs so ``len(batch) * longest_in_batch <=
        max_batch_tokens`` -> roughly constant memory per batch regardless of
        document lengths (short-doc batches hold many docs, long-doc batches
        few). A single doc longer than the budget gets its own batch.
      - otherwise: fixed ``batch_size`` docs per batch (still length-sorted).

    Returns a list of index lists, indices referring to the original sequence.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    if max_batch_tokens is None:
        return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = lengths[i]
        if cur and max(cur_max, L) * (len(cur) + 1) > max_batch_tokens:
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, L)
    if cur:
        batches.append(cur)
    return batches


def extract_norm_lora_emb_batch(model, ctx_tokenizer, texts, max_length=2048,
                                batch_size=8, sort_by_length=True, max_batch_tokens=None,
                                to_cpu=False):
    """Extract norm_lora_emb for multiple documents in batches.

    Hooks on model.hypernet.head and captures input[0] (the L2-normalised
    vectors just before the LoRA head) for a full batch in one forward pass.

    Documents are tokenized once, grouped into length-sorted batches (and
    optionally packed to a token budget via ``max_batch_tokens``; see
    :func:`_make_batches`), embedded, then returned **in the original input
    order**. Padding is per batch (to that batch's longest doc, <= max_length)
    and attention-masked, so length-sorting changes only speed and peak memory,
    not the returned vectors.

    Args:
        model: ModulatedPretrainedModel with generate_weights method
        ctx_tokenizer: context encoder tokenizer
        texts: list of document strings
        max_length: max token length per document (truncated if exceeded)
        batch_size: docs per batch when ``max_batch_tokens`` is None
        sort_by_length: group similar-length docs together (default True). Set
            False for the legacy contiguous batching (kept for comparison).
        max_batch_tokens: if set, use token-budget batching (variable docs per
            batch, ~constant memory) instead of a fixed ``batch_size``.
        to_cpu: move each captured adapter to CPU before returning (default
            False). Each adapter is a ~147k-float tensor; for large batches
            keeping them all on the GPU pins VRAM, so pass ``to_cpu=True`` when
            embedding many documents and copy back to the device on demand.

    Returns:
        list of torch.Tensor, each of shape [n_layers, n_modules, r, latent_size],
        aligned to ``texts`` (on CPU if ``to_cpu``, else on ``model.device``).
    """
    token_lists = [_tokenize_single(ctx_tokenizer, t, max_length) for t in texts]
    lengths = [int(t.shape[0]) for t in token_lists]

    if sort_by_length or max_batch_tokens is not None:
        batches = _make_batches(lengths, batch_size=batch_size, max_batch_tokens=max_batch_tokens)
    else:
        batches = [list(range(i, min(i + batch_size, len(texts))))
                   for i in range(0, len(texts), batch_size)]

    out = [None] * len(texts)
    for idxs in batches:
        ctx_ids, ctx_attn_mask = _pad_and_stack([token_lists[i] for i in idxs], device=model.device)

        captured = {}

        def hook_fn(module, input, output):
            captured["emb"] = input[0].detach()

        hook = model.hypernet.head.register_forward_hook(hook_fn)

        try:
            with torch.no_grad():
                model.generate_weights(ctx_ids, ctx_attn_mask)
        finally:
            hook.remove()
            _safe_reset(model)

        # Shape: [bs, n_layers, n_modules, r, latent_size]
        emb = captured["emb"]
        for j, orig_i in enumerate(idxs):
            out[orig_i] = emb[j].cpu() if to_cpu else emb[j]  # scatter back to original position

    return out


def mean_pool_and_flatten(adapter):
    """Pool an adapter over the module and rank axes and flatten the layers.

    Turns a ``norm_lora_emb`` adapter into the flat "pooled adapter" vector that the
    released ``*_norm_lora_emb.npz`` files store and that the cosine-similarity
    benchmarks consume.

    Args:
        adapter: torch.Tensor or np.ndarray of shape [n_layers, n_modules, r, latent]
            (as returned by ``extract_norm_lora_emb``).

    Returns:
        np.ndarray of shape [n_layers * latent] as fp16.
    """
    if isinstance(adapter, np.ndarray):
        adapter = torch.from_numpy(adapter)
    pooled = adapter.mean(dim=(1, 2))  # [n_layers, latent]
    return pooled.flatten().cpu().to(torch.float16).numpy()
