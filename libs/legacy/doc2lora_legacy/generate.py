"""Text generation with injected doc2lora latents."""

import torch
from einops import rearrange, unpack

from ctx_to_lora.modeling.lora_merger import combine_lora
from ctx_to_lora.modeling.lora_layer import apply_lora_to_layers


def internalize_from_latents(model, encoder_latents):
    """Inject pre-computed perceiver encoder latents into the model as LoRA weights.

    This runs the latents through the perceiver decoder and HyperLoRA pipeline,
    then applies the resulting LoRA weights to the base model. After calling this,
    the model is conditioned on the document and ready for generation.

    Args:
        model: ModulatedPretrainedModel
        encoder_latents: torch.Tensor of shape [n_layers, n_latents, hidden_dim]
            (typically [26, 8, 512]). Will be moved to model device if needed.

    Note:
        Call model.reset() before this to clear any previously internalized document.
    """
    if not isinstance(encoder_latents, torch.Tensor):
        encoder_latents = torch.from_numpy(encoder_latents)
    encoder_latents = encoder_latents.to(model.device)

    model.patch_lora_forward()
    perceiver = model.hypernet.aggregator.perceiver
    aggregator = model.hypernet.aggregator
    hypernet = model.hypernet

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # Decoder: latents -> output queries
        n_latents = perceiver.encoder.n_latents
        latent_position_ids = torch.arange(
            n_latents, device=encoder_latents.device
        ).unsqueeze(0)
        num_layers_times_bs = encoder_latents.shape[0]
        latent_position_ids = torch.tile(latent_position_ids, (1, num_layers_times_bs))

        decoder_output = perceiver.decoder(
            encoder_latents, position_ids=latent_position_ids
        )

        # Aggregator post-processing
        x = decoder_output
        per_layer_size = (
            aggregator.num_modules * aggregator.r + aggregator.num_extra_modules
        )
        x = rearrange(
            x,
            "(num_layers bs) (per_layer_sz) d -> bs (num_layers per_layer_sz) d",
            num_layers=aggregator.num_layers,
            per_layer_sz=per_layer_size,
        )
        lora_x, extra_x = unpack(
            x,
            [
                [aggregator.num_layers * aggregator.num_modules * aggregator.r],
                [aggregator.num_layers * aggregator.num_extra_modules],
            ],
            "bs * feature_dim",
        )
        lora_x = rearrange(
            lora_x,
            "bs (n_layers n_modules r) d -> bs n_layers n_modules r d",
            n_modules=aggregator.num_modules,
            n_layers=aggregator.num_layers,
            r=aggregator.r,
        )
        if not aggregator.per_rank_gen:
            lora_x = lora_x.squeeze(3)
        lora_emb = lora_x

        # HyperLoRA: lora_emb -> LoRA weights
        lora_emb = hypernet.layers(lora_emb)
        norm = torch.norm(lora_emb, dim=-1, keepdim=True)
        norm_lora_emb = lora_emb / norm
        flat_loras = hypernet.head(norm_lora_emb)

        lora_dict = hypernet._to_lora_dict(flat_loras)

    # Combine and apply LoRA to the base model
    n_ctx_chunks = torch.tensor((1,), device=model.device)
    generated_loras = combine_lora(
        lora_dict,
        n_ctx_chunks,
        lora_bias=hypernet.get_head_bias() if hypernet.config.use_bias else None,
    )
    n_queries = torch.ones(1, dtype=torch.int32, device=model.device)
    apply_lora_to_layers(
        model.base_model, hypernet.layer_indices, generated_loras, n_queries, None
    )
    model.generated_loras = lora_dict


def generate_text(model, tokenizer, prompt, max_new_tokens=256, do_sample=False, **gen_kwargs):
    """Generate text from the model after internalization.

    Args:
        model: ModulatedPretrainedModel (with LoRA applied via internalize_from_latents)
        tokenizer: base LLM tokenizer (not the context tokenizer)
        prompt: user prompt / question
        max_new_tokens: maximum tokens to generate
        do_sample: if False (default), greedy decoding -> deterministic/reproducible
            output. Passed explicitly so it overrides any sampling defaults baked
            into the base model's generation_config (e.g. Qwen3's do_sample=True,
            temperature=0.7). Set True to restore stochastic sampling.
        **gen_kwargs: forwarded to ``model.generate`` (e.g. temperature, top_p).

    Returns:
        str: generated response text
    """
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
