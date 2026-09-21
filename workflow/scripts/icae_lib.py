"""Shared ICAE helpers: load the fine-tuned ICAE model and compress documents into memory slots.

ICAE compresses one document (<=512 tokens here) into `mem_size`=128 memory slots of dim 4096
(Mistral-7B hidden). The stock model._compress is batch-1 only; `compress_batch` below replicates
the single-segment path with LEFT padding + explicit position_ids so many documents run per forward.
Validated to match the reference _compress (cos > 0.999) in tests.

A paper's benchmark vector = mean over the 128 slots -> 4096-dim (the standard ICAE embedding).
The invertible per-token adapter (kron.KronAdapter, L=128 d=4096) operates on the slots BEFORE pooling.
"""
import sys
from pathlib import Path

import torch

MAX_LEN = 512   # tokens per doc; 512 / (mem 128 * rate 4) = 1 segment -> exactly 128 slots


def load_icae(weights, code_dir, base_model, device="cuda"):
    """Load the fine-tuned ICAE (Mistral-7B) in eval mode. Mirrors simplex/decode_icae.load_icae."""
    sys.path.insert(0, str(code_dir))
    from modeling_icae_multi_span import ICAE, ModelArguments, TrainingArguments
    from peft import LoraConfig
    from safetensors.torch import load_file
    margs = ModelArguments(model_name_or_path=base_model, lora_r=512, lora_dropout=0.05, train=False)
    targs = TrainingArguments(output_dir=str(weights), model_max_length=5120,
                              fixed_mem_size=128, mean_compression_rate=4, bf16=True)
    lcfg = LoraConfig(r=512, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = ICAE(margs, targs, lcfg)
    model.load_state_dict(load_file(str(weights)), strict=False)
    return model.to(device).eval()


@torch.no_grad()
def compress_one(model, text, device):
    """Reference single-doc compress -> [128, 4096] (for validation)."""
    ids = model.tokenizer(text, truncation=True, max_length=MAX_LEN, padding=False,
                          return_attention_mask=False)["input_ids"]
    slots = model._compress(torch.LongTensor([ids]).to(device))
    return (slots.squeeze(0) if slots.dim() == 3 else slots).float().to(device)


@torch.no_grad()
def compress_batch(model, texts, device):
    """Batched single-segment compress with LEFT padding. Returns [B, 128, 4096] float32.

    Layout per row: [PAD ... PAD | text tokens | 128 memory tokens]. Memory tokens always occupy
    the last 128 columns, so slots = last_hidden[:, -mem:, :]. Left padding keeps the text adjacent
    to the memory tokens; position_ids are recomputed from the attention mask so RoPE is correct.
    """
    mem = model.mem_size
    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = model.tokenizer.eos_token_id
    enc = [model.tokenizer(t, truncation=True, max_length=MAX_LEN, padding=False,
                           return_attention_mask=False)["input_ids"] for t in texts]
    T = max(len(x) for x in enc)
    B = len(enc)
    base = model.icae.get_base_model().model
    embed = base.embed_tokens

    # build left-padded text id matrix + text attention mask
    txt_ids = torch.full((B, T), pad_id, dtype=torch.long, device=device)
    txt_mask = torch.zeros((B, T), dtype=torch.long, device=device)
    for i, x in enumerate(enc):
        L = len(x)
        txt_ids[i, T - L:] = torch.tensor(x, device=device)
        txt_mask[i, T - L:] = 1

    mem_ids = model.append_sequence.to(device).expand(B, mem)          # [B, mem] memory-token ids
    mem_mask = torch.ones((B, mem), dtype=torch.long, device=device)

    txt_emb = embed(txt_ids)
    mem_emb = model.memory_token_embed(mem_ids - model.vocab_size).to(txt_emb)
    inp = torch.cat([txt_emb, mem_emb], dim=1)                         # [B, T+mem, d]
    attn = torch.cat([txt_mask, mem_mask], dim=1)                      # [B, T+mem]
    pos = (attn.cumsum(-1) - 1).clamp(min=0)                           # left-pad-correct RoPE

    # Call the base TRANSFORMER directly (LoRA still applied): returns last_hidden_state without the
    # 32k-vocab lm_head and without materializing every layer -> ~2x faster than the CausalLM forward.
    out = base(inputs_embeds=inp, attention_mask=attn, position_ids=pos)
    h = out.last_hidden_state                                          # [B, T+mem, d]
    return h[:, -mem:, :].float()                                      # [B, mem, d]
