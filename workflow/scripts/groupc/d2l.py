"""Shared Doc2LoRA helpers for the Group C experiments (issue #142).

Every GPU script in workflow/scripts/groupc/ that has to embed or decode goes through here, so the
extraction settings, the internalisation path and the label post-processing are identical across
experiments (#95 decode fidelity, #101 incoherent-cluster control, #94 adapted-space decode,
#73 prompt sensitivity, #74 label calibration, #105 stability).

Requires `ctx_to_lora` on the path (the doc-to-lora source tree, see config `doc_to_lora_src`) and a
Doc2LoRA hypernetwork checkpoint (config `qwen_checkpoint_path`, or $DOC2LORA_CKPT).
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Qwen3-4B Doc2LoRA tensor geometry: 36 layers x 1 module x rank 8 x latent 512.
L, R, LAT = 36, 8, 512

# The verbatim 2-3 word field prompt behind every Doc2LoRA cluster label in the manuscript
# (exps/2026-06-11-baseline-trees/decode_fullrank_field23.py).
FIELD23_PROMPT = (
    "In 2 to 3 words, name the scientific field that all of these documents "
    "belong to. Reply with only the field name."
)


def _add_paths() -> None:
    for p in ("libs/legacy", "libs/doc2lora"):
        ap = os.path.abspath(p)
        if os.path.isdir(ap) and ap not in sys.path:
            sys.path.insert(0, ap)
    src = os.environ.get("DOC_TO_LORA_SRC")
    if src and os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)


def _ctx_to_sdpa(model) -> None:
    """Switch the CONTEXT encoder to SDPA attention so batched extraction works.

    The legacy batched extractor right-pads (doc2lora_legacy.embed._pad_and_stack), and Qwen3 under
    flash_attention_2 refuses a right-padded batch outright ("call tokenizer.padding_side = 'left'").
    SDPA honours the attention mask on either side, and batched extraction under SDPA reproduces the
    unpadded single-document extraction to cos 0.9999 (verified on Qwen3-4B), so this changes only
    the kernel, not the vectors.  The generator half keeps its original attention implementation.
    """
    enc = getattr(model, "ctx_encoder", None)
    if enc is None:
        return
    for mod in [enc] + list(enc.modules()):
        cfg = getattr(mod, "config", None)
        if cfg is not None and getattr(cfg, "_attn_implementation", None) == "flash_attention_2":
            cfg._attn_implementation = "sdpa"


def load(ckpt: str | None = None, mode: str = "full", ctx_sdpa: bool = True):
    """Load a Doc2LoRA checkpoint. mode: 'full' (embed+generate) | 'embed' | 'generate'."""
    _add_paths()
    from doc2lora_legacy import load_model

    ckpt = ckpt or os.environ.get("DOC2LORA_CKPT")
    if not ckpt:
        raise SystemExit("no checkpoint: pass ckpt= or set $DOC2LORA_CKPT")
    print(f"[d2l] loading {ckpt} (mode={mode})", flush=True)
    out = load_model(str(ckpt), mode=mode)
    if ctx_sdpa:
        _ctx_to_sdpa(out[0])
    return out


def extract_full(model, ctx_tok, texts, max_batch_tokens: int = 6144, max_length: int = 2048):
    """Full-rank norm_lora_emb per text -> list of tensors [L, 1, R, LAT] (on model device)."""
    _add_paths()
    from doc2lora_legacy.embed import extract_norm_lora_emb_batch

    return extract_norm_lora_emb_batch(model, ctx_tok, list(texts),
                                       max_length=max_length,
                                       max_batch_tokens=max_batch_tokens)


def full_mean(model, ctx_tok, texts, chunk: int = 256, max_batch_tokens: int = 6144) -> np.ndarray:
    """Mean full-rank embedding over `texts` -> [L, R, LAT] float32 (float64 accumulation)."""
    acc = torch.zeros(L, 1, R, LAT, dtype=torch.float64)
    n = 0
    for c0 in range(0, len(texts), chunk):
        batch = list(texts[c0:c0 + chunk])
        embs = extract_full(model, ctx_tok, batch, max_batch_tokens=max_batch_tokens)
        acc += torch.stack([e.detach().cpu() for e in embs]).double().sum(0)
        n += len(batch)
    return (acc / max(n, 1)).squeeze(1).float().numpy()


def decode(model, gen_tok, tensor, prompt: str = FIELD23_PROMPT, max_new_tokens: int = 16) -> str:
    """Internalise a full-rank tensor ([L,R,LAT] or [L,1,R,LAT]) and greedily decode `prompt`."""
    _add_paths()
    from doc2lora_legacy import generate_text, internalize_from_norm_lora_emb

    t = (tensor.detach().float().cpu() if torch.is_tensor(tensor)
         else torch.as_tensor(np.asarray(tensor, dtype=np.float32)))
    t = t.reshape(L, 1, R, LAT).contiguous().to(model.device)
    model.reset()
    internalize_from_norm_lora_emb(model, t)
    try:
        out = generate_text(model, gen_tok, prompt, max_new_tokens=max_new_tokens)
    finally:
        model.reset()
    return out


def normalize_label(text: str) -> str:
    """Same label post-processing as decode_fullrank_field23.py (first line, strip prefixes)."""
    lines = [ln.strip(" \t\"'*`-") for ln in text.splitlines() if ln.strip()]
    lab = lines[0] if lines else text.strip()
    lab = re.sub(r"^(cluster label|cluster|label|topic|field)\s*[:\-]\s*", "", lab, flags=re.I)
    return lab.strip(" \t\"'*`.")


def aps_texts(path: str = "data/aps/paper_text.parquet") -> dict[int, str]:
    """paper_id -> text (title + abstract) for the APS corpus."""
    df = pd.read_parquet(path)
    idc = "aps_paper_id" if "aps_paper_id" in df.columns else "paper_id"
    return dict(zip(df[idc].astype(int), df["text"].astype(str)))


def aps_frame(path: str = "data/aps/paper_text.parquet") -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.rename(columns={"aps_paper_id": "paper_id"})
