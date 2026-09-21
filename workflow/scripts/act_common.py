"""Raw-LLM-activation arm: extract hidden states and decode them by patching (#152).

The comparison this supports: is the \\doctolora gene more decodable than the base
model's own internal activations? Text encoders cannot answer that question -- they
have no decoder -- but activations can, because the model itself decodes them.

Both arms run on **Qwen3-4B-Instruct-2507**, the base the doc2lora qwen checkpoint
decodes through, so generator, tokenizer and instruction text are identical and the
only thing that differs is the injected object.

Decoder (training-free, Patchscopes / SelfIE style)
---------------------------------------------------
Read a hidden state from the document's forward pass at layer ``l``, then run a
*separate* decode prompt and overwrite the hidden state at a placeholder position with
that vector, at the same layer ``l``. Read and write happen at the same interface --
the INPUT to block ``l``, i.e. ``output_hidden_states[l]`` -- so no representational
offset is introduced. No training anywhere, which is what makes it comparable to
doc2lora's decode.

Index convention (HF): ``hidden_states[0]`` is the embedding output and
``hidden_states[i]`` is the output of block ``i-1`` = the input of block ``i``. A
``forward_pre_hook`` on ``model.model.layers[l]`` therefore writes exactly what
``hidden_states[l]`` reads.

Representations (``act_k36`` dropped by decision 2026-09-18):
  ``act_mean``  mean over non-pad token positions   [2560]
  ``act_last``  the last non-pad position           [2560]

Known asymmetry, to be stated in the manuscript: the activation arm needs a placeholder
token in the decode prompt to patch into, the gene arm does not. Both arms get the same
instruction text; only the placeholder differs. And the objects are not the same size --
a gene is 147,456 floats (36 x 8 x 512), an activation vector is 2,560 -- so the
activation arm is the smaller object by construction. Read a gene win with that in mind;
an activation win would be the stronger result.

    CUDA_VISIBLE_DEVICES=0 python act_common.py --smoke
"""
import argparse
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

QWEN_ID = "Qwen/Qwen3-4B-Instruct-2507"
# the placeholder the decode prompt patches into; content is irrelevant because its
# hidden state is overwritten, but it must survive tokenization as its own position
PLACEHOLDER = "?"
DOC_PREFIX = "Document: "
TOPIC_PROMPT = "In one sentence, what is the specific topic of this document?"


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def load_qwen(device="cuda", dtype=torch.bfloat16, model_id=QWEN_ID):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    return dict(model=model, tok=tok, device=device,
                n_layers=model.config.num_hidden_layers,
                d=model.config.hidden_size)


def _blocks(model):
    return model.model.layers


# --------------------------------------------------------------------------- #
# read: extract activations
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_acts(Q, texts, layer, mode="mean", batch_size=8, max_length=1024):
    """Layer-`layer` hidden states for each text -> [N, d] float32.

    `layer` indexes `output_hidden_states`: 0 = embeddings, l = input of block l.
    Padding is masked out of both pooling modes.
    """
    assert mode in ("mean", "last")
    model, tok = Q["model"], Q["tok"]
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = list(texts[i:i + batch_size])
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length).to(Q["device"])
        hs = model(**enc, output_hidden_states=True).hidden_states[layer]  # [B, T, d]
        m = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)               # [B, T, 1]
        if mode == "mean":
            v = (hs * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            idx = enc["attention_mask"].sum(1) - 1                          # last real token
            v = hs[torch.arange(hs.shape[0], device=hs.device), idx]
        out.append(v.float().cpu())
    return torch.cat(out, 0)


# --------------------------------------------------------------------------- #
# write: patch a vector in and decode
# --------------------------------------------------------------------------- #
@contextmanager
def _patch_at(model, layer, positions, vectors):
    """Overwrite block-`layer` INPUT hidden states at `positions` with `vectors`.

    `vectors` is [n_pos, d] (the same injection for every batch row) or [B, n_pos, d]
    (one per row) -- the batched form lets a whole batch of documents share one decode
    prompt, since only the injected object differs between them.

    Only fires on the prefill pass (seq len > 1); decode steps pass a single token and
    must be left alone, or every generated token would be overwritten too.
    """
    pos = torch.as_tensor(positions, dtype=torch.long)
    vec = vectors if isinstance(vectors, torch.Tensor) else torch.stack(list(vectors))
    if vec.dim() == 1:
        vec = vec.reshape(1, -1)
    if vec.dim() == 2:
        vec = vec.reshape(len(pos), -1).unsqueeze(0)     # -> [1, n_pos, d], broadcasts
    assert vec.shape[-2] == len(pos), f"{tuple(vec.shape)} vs {len(pos)} positions"

    def pre_hook(_mod, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        if h.shape[1] <= 1:                     # generation step, not prefill
            return None
        h = h.clone()
        h[:, pos.to(h.device), :] = vec.to(device=h.device, dtype=h.dtype)
        if args:
            return (h,) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = h
        return args, kwargs

    handle = _blocks(model)[layer].register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


def build_decode_inputs(Q, prompt, n_slots=1):
    """Chat-formatted decode prompt carrying exactly `n_slots` patchable positions.

    Built by CONCATENATING TOKEN IDS rather than by locating a marker in the decoded
    string: BPE merges the placeholder with neighbouring characters (Qwen turns
    "? \\n\\n" into a single 'Ġ?ĊĊ' token), so any string-offset arithmetic silently
    points at the wrong position -- the first version of this patched the token 'In'.
    Concatenation makes the indices exact by construction.

    Returns (input_ids [1, T], placeholder positions [n_slots]).
    """
    tok = Q["tok"]
    user = f"{DOC_PREFIX}{PLACEHOLDER}\n\n{prompt}"
    full = tok.apply_chat_template([{"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)
    head, tail = full.split(PLACEHOLDER, 1)
    head_ids = tok(head, add_special_tokens=False)["input_ids"]
    tail_ids = tok(tail, add_special_tokens=False)["input_ids"]
    slot_id = tok(PLACEHOLDER, add_special_tokens=False)["input_ids"][0]
    slot_ids = [slot_id] * n_slots
    ids = torch.tensor([head_ids + slot_ids + tail_ids], dtype=torch.long)
    positions = list(range(len(head_ids), len(head_ids) + n_slots))
    return ids, positions


@torch.no_grad()
def patch_decode(Q, vec, layer, prompt=TOPIC_PROMPT, max_new_tokens=64):
    """Decode a single activation vector by patching it into the decode prompt."""
    model = Q["model"]
    ids, positions = build_decode_inputs(Q, prompt, n_slots=1)
    ids = ids.to(Q["device"])
    v = torch.as_tensor(vec).reshape(1, -1)
    with _patch_at(model, layer, positions, v):
        out = model.generate(input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=Q["tok"].eos_token_id)
    return Q["tok"].decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def nopatch_decode(Q, prompt=TOPIC_PROMPT, max_new_tokens=64):
    """The same prompt with NOTHING injected -- the floor every arm must clear.

    Whatever this produces is what the prompt alone hallucinates; an arm that does not
    beat it is not carrying document information.
    """
    ids, _ = build_decode_inputs(Q, prompt, n_slots=1)
    ids = ids.to(Q["device"])
    out = Q["model"].generate(input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False,
                              pad_token_id=Q["tok"].eos_token_id)
    return Q["tok"].decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# corpus
# --------------------------------------------------------------------------- #
def load_aps_text(n=None, seed=0, min_abstract=200):
    """title + abstract, the verbatim form both arms receive."""
    import pandas as pd

    df = pd.read_parquet(ROOT / "data/aps/paper_text_pid.parquet",
                         columns=["paper_id", "title", "abstract"])
    df = df[df.abstract.notna() & (df.abstract.str.len() > min_abstract)]
    if n is not None:
        df = df.sample(n=min(n, len(df)), random_state=seed)
    df = df.assign(title=df.title.fillna("").str.strip())
    df = df.assign(text=(df.title + ". " + df.abstract.str.strip()).str.strip(". "))
    return df[["paper_id", "title", "abstract", "text"]].reset_index(drop=True)


# --------------------------------------------------------------------------- #
def _smoke():
    Q = load_qwen()
    print(f"[load] {QWEN_ID}  layers={Q['n_layers']}  d={Q['d']}")
    docs = load_aps_text(n=3, seed=1)
    ids, pos = build_decode_inputs(Q, TOPIC_PROMPT)
    print(f"[prompt] {ids.shape[1]} tokens, placeholder at {pos}")
    print(f"[floor/no patch] {nopatch_decode(Q)}")
    for layer in (8, 18):
        A = extract_acts(Q, list(docs.text), layer, mode="mean")
        print(f"\n[layer {layer}] acts {tuple(A.shape)}  "
              f"norm {A.norm(dim=1).mean():.2f}")
        for i in range(len(docs)):
            print(f"  {docs.title[i][:60]}\n    -> {patch_decode(Q, A[i], layer)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        _smoke()
    else:
        print(__doc__)
