"""Shared Text-to-LoRA (T2L) primitives for the hypernetwork-adapter baseline (issue #151).

T2L (Charakorn et al., ICML 2025) generates a LoRA adapter from a *task description*:

    text --gte-large-en-v1.5 (frozen)--> z0 (1024) --TaskEncoder--> z1 (64)
         --mixer/mlp1-3 + layer&module embeddings--> heads --> (A, B) --> dW = B @ A

The released Mistral checkpoint targets ``q_proj``/``v_proj`` on all 32 layers of
``mistralai/Mistral-7B-Instruct-v0.2`` at r=8, alpha=16, use_rslora=True -- the SAME
base model as the ICAE arm (workflow/config.yaml:icae_base_model), so decodes are
comparable at the generator level.

Three candidate "embedding" positions are exposed, per the issue:

  E0  the frozen gte-large output              (1024-d)   -- an INPUT, not a learned layer
  E1  TaskEncoder output = Linear(1024,64)+LN  (64-d)     -- the only document-dependent
                                                             learned representation in T2L
  E2  the generated LoRA factors (A, B)        (3.41M)    -- the adapter itself

Averaging convention (E2). Doc2LoRA's head is LINEAR and emits both LoRA factors, so
averaging doc2lora embeddings averages A and B separately and the mean stays rank-8
(libs/doc2lora/doc2lora/arithmetic.py:internalize_from_norm_lora_emb). ``mean_lora_sd``
below does exactly that, which is the structural analogue. The alternative -- the true
mean of the dW matrices, mean(B_i @ A_i) -- is a DIFFERENT operation for both methods
(rank 8N) and is deliberately out of scope here.

Everything runs in the main env (transformers 4.51.3): hypermod.pt is a plain state_dict,
so the T2L pin (transformers==4.46.2) is not required. Only ``inflect``/``torchmetrics``/
``wandb`` had to be installed for hyper_llm_modulator's import chain.

    export PYTHONPATH=$T2L_SRC
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
T2L_SRC = os.environ.get("T2L_SRC", str(ROOT / "text-to-lora" / "src"))
HYPERMOD_DIR = os.environ.get(
    "T2L_HYPERMOD_DIR",
    str(ROOT / "data/agent_assets/hf_cache/hub/models--SakanaAI--text-to-lora/snapshots"
        "/6e571eda2188b216f027263cd28c99c0fdcf2fa3/trained_t2l/mistral_7b_t2l"),
)
os.environ.setdefault("HF_HOME", str(ROOT / "data/agent_assets/hf_cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if T2L_SRC not in sys.path:
    sys.path.insert(0, T2L_SRC)


# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #
class _chdir:
    """contextlib.chdir backport (this env is Python 3.10)."""

    def __init__(self, path):
        self.path, self._prev = str(path), None

    def __enter__(self):
        self._prev = os.getcwd()
        os.chdir(self.path)
        return self

    def __exit__(self, *exc):
        os.chdir(self._prev)
        return False



def load_t2l(device="cuda", hypermod_dir=HYPERMOD_DIR):
    """Load the T2L hypernetwork, its gte encoder, and the base Mistral PeftModel.

    T2L's ``get_tokenizer`` resolves ``chat_templates/<model>/chat_template.jinja``
    relative to the CWD, so the load runs from the T2L repo root -- we decode with
    THEIR chat template, not a substitute of ours.
    """
    from hyper_llm_modulator.hyper_modulator import load_hypermod_checkpoint
    from hyper_llm_modulator.utils import get_layers

    with _chdir(Path(T2L_SRC).parent):
        (args, hypermod, model, tokenizer, emb_model, emb_tokenizer,
         fmt_fn, pool_fn) = load_hypermod_checkpoint(f"{hypermod_dir}/hypermod.pt", device)
    layer_indices = torch.tensor(range(len(get_layers(model))), dtype=torch.long, device=device)
    # generation does not need hidden states; load_hypermod_checkpoint turns them on
    model.config.output_hidden_states = False
    return dict(args=args, hypermod=hypermod, model=model, tokenizer=tokenizer,
                emb_model=emb_model, emb_tokenizer=emb_tokenizer, fmt_fn=fmt_fn,
                pool_fn=pool_fn, layer_indices=layer_indices, device=device,
                hypermod_dir=hypermod_dir)


# --------------------------------------------------------------------------- #
# E0 / E1 / E2
# --------------------------------------------------------------------------- #
@torch.no_grad()
def embed_e0(T, texts, batch_size=16):
    """E0: frozen gte-large-en-v1.5 CLS embedding [N, 1024] (fp32, un-normalized)."""
    from hyper_llm_modulator.utils import embed_texts

    v = embed_texts(list(texts), T["emb_model"], T["emb_tokenizer"], T["fmt_fn"],
                    T["pool_fn"], T["device"], batch_size=batch_size)
    return v.float()


@torch.no_grad()
def e0_to_e1(T, e0):
    """E1: TaskEncoder(E0) = Linear(1024, 64) + LayerNorm -> [N, 64]."""
    out = T["hypermod"].task_encoder(e0.to(T["device"]))
    return out["encoded_task_emb"].detach().float()


@torch.no_grad()
def e1_to_lora(T, e1_row):
    """E2: one E1 vector [64] (or [1, 64]) -> LoRA state dict of A/B factors (on CPU)."""
    e1_row = e1_row.reshape(1, -1).to(T["device"])
    return T["hypermod"].gen_lora(T["layer_indices"], e1_row)


def text_to_lora(T, text):
    """Convenience: document text -> (e0, e1, lora_state_dict)."""
    e0 = embed_e0(T, [text])
    e1 = e0_to_e1(T, e0)
    return e0[0].cpu(), e1[0].cpu(), e1_to_lora(T, e1[0])


# --------------------------------------------------------------------------- #
# adapter arithmetic (factor-space, matching doc2lora's linear head)
# --------------------------------------------------------------------------- #
def mean_lora_sd(sds, weights=None):
    """Weighted mean of LoRA state dicts, averaging A and B separately (rank stays 8).

    This mirrors doc2lora: its head is linear, so averaging embeddings averages the
    emitted factors. Not the same as mean(B_i @ A_i) -- see module docstring.
    """
    if weights is None:
        weights = [1.0 / len(sds)] * len(sds)
    assert len(weights) == len(sds)
    out = {}
    for k in sds[0]:
        acc = torch.zeros_like(sds[0][k], dtype=torch.float32)
        for w, sd in zip(weights, sds):
            acc += float(w) * sd[k].float()
        out[k] = acc.to(sds[0][k].dtype)
    return out


class LoraAccumulator:
    """Streaming fp32 mean of LoRA state dicts.

    Per-document adapters are 6.5 MiB in fp16 (3.41M params); the label arm touches
    up to 2,000 members x 28 nodes, so they are never stored -- only accumulated.
    """

    def __init__(self):
        self.acc = None
        self.n = 0

    def add(self, sd, weight=1.0):
        if self.acc is None:
            self.acc = {k: torch.zeros_like(v, dtype=torch.float32, device="cpu") for k, v in sd.items()}
        for k, v in sd.items():
            self.acc[k] += float(weight) * v.float().cpu()
        self.n += weight
        return self

    def mean(self, dtype=torch.float32):
        assert self.n > 0, "nothing accumulated"
        return {k: (v / self.n).to(dtype) for k, v in self.acc.items()}


def _pair_AB(sd):
    """Group a LoRA state dict into (A, B) pairs keyed by the shared module prefix."""
    pairs = {}
    for k, v in sd.items():
        if "lora_A" in k:
            pairs.setdefault(k.replace("lora_A", "*"), {})["A"] = v
        elif "lora_B" in k:
            pairs.setdefault(k.replace("lora_B", "*"), {})["B"] = v
    return pairs


def dw_inner(sd1, sd2):
    """Frobenius inner product <dW1, dW2> summed over all layers/modules.

    Uses <B1 A1, B2 A2>_F = tr((B1^T B2)(A2 A1^T)) so the 671M-parameter dense dW is
    never materialized -- only [r, r] matrices (r=8).
    """
    p1, p2 = _pair_AB(sd1), _pair_AB(sd2)
    tot = 0.0
    for key, ab1 in p1.items():
        ab2 = p2[key]
        A1, B1 = ab1["A"].float(), ab1["B"].float()
        A2, B2 = ab2["A"].float(), ab2["B"].float()
        # peft layout: A [r, in], B [out, r]
        tot += float(torch.trace((B1.T @ B2) @ (A2 @ A1.T)))
    return tot


def dw_cos(sd1, sd2):
    """Cosine similarity between two generated dW's (exact, never materializes dW)."""
    n1 = dw_inner(sd1, sd1) ** 0.5
    n2 = dw_inner(sd2, sd2) ** 0.5
    if n1 == 0 or n2 == 0:
        return float("nan")
    return dw_inner(sd1, sd2) / (n1 * n2)


def dw_norm(sd):
    return dw_inner(sd, sd) ** 0.5


# --------------------------------------------------------------------------- #
# decode
# --------------------------------------------------------------------------- #
def apply_lora(T, lora_sd):
    """Load a generated LoRA state dict into the base PeftModel."""
    from peft import set_peft_model_state_dict

    model = T["model"]
    dev, dt = model.device, next(model.parameters()).dtype
    sd = {k: v.to(device=dev, dtype=dt) for k, v in lora_sd.items()}
    info = set_peft_model_state_dict(model, sd)
    return info


def build_prompt(T, user_content, system_message="", assistant_prefill=""):
    """Reproduce T2L's training-time prompt exactly.

    `preprocessing.get_prompt_formatting_fn`'s `f_intx` branch builds
    ``[{"role": "system", ...}, {"role": "user", ...}]`` and appends the assistant
    prefill, and every task metadata carries ``system_message: ''``. Their chat
    template emits ``bos_token`` itself when a system message is present, so the
    result is tokenized with ``add_special_tokens=False`` to avoid a second BOS.
    """
    tok = T["tokenizer"]
    chat = [{"role": "system", "content": system_message},
            {"role": "user", "content": user_content}]
    text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    return text + assistant_prefill


@torch.no_grad()
def decode(T, prompt, max_new_tokens=64, do_sample=False, assistant_prefill=""):
    """Greedy-decode the currently applied adapter, in T2L's own prompt format."""
    tok, model = T["tokenizer"], T["model"]
    text = build_prompt(T, prompt, assistant_prefill=assistant_prefill)
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    out = model.generate(input_ids=ids, max_new_tokens=max_new_tokens, do_sample=do_sample,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def decode_lora(T, lora_sd, prompt, max_new_tokens=64):
    apply_lora(T, lora_sd)
    return decode(T, prompt, max_new_tokens=max_new_tokens)


# --------------------------------------------------------------------------- #
# corpus helpers
# --------------------------------------------------------------------------- #
def load_aps_text(paper_ids=None, n=None, seed=0):
    """title + abstract for APS papers, in the verbatim form every arm receives."""
    import pandas as pd

    df = pd.read_parquet(ROOT / "data/aps/paper_text_pid.parquet",
                         columns=["paper_id", "title", "abstract"])
    if paper_ids is not None:
        df = df[df.paper_id.isin(set(int(p) for p in paper_ids))]
    df = df[df.abstract.notna() & (df.abstract.str.len() > 200)]
    if n is not None:
        df = df.sample(n=min(n, len(df)), random_state=seed)
    df = df.assign(title=df.title.fillna("").str.strip())
    df = df.assign(text=(df.title + ". " + df.abstract.str.strip()).str.strip(". "))
    return df[["paper_id", "title", "abstract", "text"]].reset_index(drop=True)
