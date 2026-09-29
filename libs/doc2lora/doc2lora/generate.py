"""Text generation conditioned on a doc2lora adapter."""

import torch

from doc2lora.arithmetic import internalize_from_norm_lora_emb


def generate_text(model, tokenizer, prompt, max_new_tokens=256, do_sample=False, **gen_kwargs):
    """Generate text from the model after an adapter has been internalized.

    Args:
        model: ModulatedPretrainedModel with an adapter already injected (e.g. via
            ``internalize_from_norm_lora_emb`` or :func:`decode_adapter`).
        tokenizer: base LLM tokenizer (not the context tokenizer).
        prompt: user prompt / question.
        max_new_tokens: maximum tokens to generate.
        do_sample: if False (default), greedy decoding -> deterministic/reproducible
            output. Passed explicitly so it overrides any sampling defaults baked
            into the base model's generation_config (e.g. Qwen3's do_sample=True,
            temperature=0.7). Set True to restore stochastic sampling.
        **gen_kwargs: forwarded to ``model.generate`` (e.g. temperature, top_p).

    Returns:
        str: generated response text.
    """
    # Under greedy decoding, neutralize the sampling-only defaults that some base
    # models bake into their generation_config (Qwen3 ships temperature=0.7,
    # top_k=20, top_p=0.8). Left as-is, transformers emits a UserWarning per param
    # complaining they're unused when do_sample=False. Only override when the
    # caller hasn't explicitly asked for them.
    if not do_sample:
        gen_kwargs.setdefault("temperature", 1.0)
        gen_kwargs.setdefault("top_k", 50)
        gen_kwargs.setdefault("top_p", 1.0)
    chat = [{"role": "user", "content": prompt}]
    chat_ids = tokenizer.apply_chat_template(
        chat,
        add_special_tokens=False,
        return_attention_mask=False,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=chat_ids, max_new_tokens=max_new_tokens,
            do_sample=do_sample, **gen_kwargs,
        )
    # Decode only the newly generated tokens (skip the input prompt)
    new_tokens = outputs[0][chat_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def decode_adapter(model, tokenizer, adapter, prompt="In one sentence, what is the specific topic of this document?",
                   *, max_new_tokens=80, **gen_kwargs):
    """Decode an adapter back into text — the one-call round-trip.

    Wraps the full conditioning cycle so you never forget the
    ``internalize -> generate -> reset`` dance (a stale adapter leaks into the next
    decode if ``model.reset()`` is skipped):

        1. inject ``adapter`` as LoRA (``internalize_from_norm_lora_emb``),
        2. answer ``prompt`` with the conditioned model (``generate_text``),
        3. ``model.reset()`` to leave the model clean for the next adapter.

    Args:
        model: ModulatedPretrainedModel.
        tokenizer: base LLM tokenizer.
        adapter: a ``norm_lora_emb`` tensor [n_layers, n_modules, r, latent] (e.g. from
            ``extract_norm_lora_emb`` or any arithmetic op on such tensors).
        prompt: what to ask about the adapter. The default asks for the topic; pass a
            different prompt to summarize, list keywords, etc. The prompt wording sets
            the *abstraction altitude* of the answer — ask for the "field" for a broad
            label, the "specific topic" for a narrow one; the adapter does not
            self-select granularity.
        max_new_tokens: generation budget.
        **gen_kwargs: forwarded to ``generate_text`` / ``model.generate``.

    Returns:
        str: the decoded text.
    """
    internalize_from_norm_lora_emb(model, adapter)
    try:
        return generate_text(model, tokenizer, prompt,
                             max_new_tokens=max_new_tokens, **gen_kwargs)
    finally:
        model.reset()


def chat_with_adapter(model, tokenizer, adapter, messages, *, max_new_tokens=256,
                      do_sample=False, **gen_kwargs):
    """Multi-turn chat with a model conditioned on an adapter.

    The multi-turn analog of :func:`decode_adapter`: it internalizes ``adapter`` and
    generates the next assistant reply given the *whole* conversation so far, then
    resets. Call it once per turn with the growing history (it is stateless across
    calls, so it composes cleanly with reactive UIs like ``marimo.ui.chat``).

    Args:
        model: ModulatedPretrainedModel.
        tokenizer: base LLM tokenizer.
        adapter: a ``norm_lora_emb`` tensor (from ``extract_norm_lora_emb`` or any
            arithmetic op on such tensors). The conversation is conditioned on it.
        messages: the conversation as a list of ``{"role", "content"}`` dicts, with
            roles ``"user"`` / ``"assistant"`` (an optional leading ``"system"``).
            The last message should be the user's latest turn.
        max_new_tokens: generation budget for the reply.
        do_sample: greedy by default (see :func:`generate_text`).
        **gen_kwargs: forwarded to ``model.generate``.

    Returns:
        str: the assistant's next reply.
    """
    # See generate_text: neutralize sampling-only defaults under greedy decoding.
    if not do_sample:
        gen_kwargs.setdefault("temperature", 1.0)
        gen_kwargs.setdefault("top_k", 50)
        gen_kwargs.setdefault("top_p", 1.0)
    internalize_from_norm_lora_emb(model, adapter)
    try:
        chat_ids = tokenizer.apply_chat_template(
            list(messages),
            add_special_tokens=False,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=chat_ids, max_new_tokens=max_new_tokens,
                do_sample=do_sample, **gen_kwargs,
            )
        new_tokens = outputs[0][chat_ids.shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)
    finally:
        model.reset()
