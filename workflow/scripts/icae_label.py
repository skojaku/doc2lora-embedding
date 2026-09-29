"""ICAE baseline labels for the comparison tree.

Mirrors the doc2lora procedure in ICAE's bottleneck: for each node, compress the
K centroid-nearest member docs into 128 memory slots each, AVERAGE the slots
across members, then decode the mean back to text and keep a short label. This
is the apples-to-apples analogue of "average the adapters, then decode the
mean." Output: icae_tree.json.

Run on aster:
  CUDA_VISIBLE_DEVICES=0 python ../2026-06-11-baseline-trees/icae_label.py
"""
import json
import os
import re
import sys
from pathlib import Path

import torch

from bench_data import out_dir, path as cfg_path   # bench_data.py sits next to this file

# ICAE is third-party: its weights and source tree are named in workflow/config.yaml.
WEIGHTS = Path(cfg_path("icae_weights"))
CODE_DIR = Path(cfg_path("icae_code_dir"))
BASE_MODEL = cfg_path("icae_base_model")
DATA = out_dir("labels")
NODES = DATA / (sys.argv[1] if len(sys.argv) > 1 else "nodes.json")
OUT = DATA / (sys.argv[2] if len(sys.argv) > 2 else "icae_tree.json")

K = 8                  # members averaged per node
MAX_LEN = 512          # per-doc token cap before compression
MAX_NEW = 28
DEVICE = "cuda"
# VERBATIM the Doc2LoRA instruction, same word budget, for a like-for-like
# comparison (shared_prompt.py). Bound to the module name LABEL_PROMPT because
# icae_abstraction_walk.py monkeypatches that attribute to sweep other prompts.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_prompt import FIELD23_PROMPT as LABEL_PROMPT  # noqa: E402

sys.path.insert(0, str(CODE_DIR))
from modeling_icae_multi_span import ICAE, ModelArguments, TrainingArguments  # noqa: E402
from peft import LoraConfig  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

INST_LEFT = torch.LongTensor([[1, 733, 16289, 28793]]).to(DEVICE)
INST_RIGHT = [733, 28748, 16289, 28793]


def load_model():
    margs = ModelArguments(model_name_or_path=BASE_MODEL, lora_r=512,
                           lora_dropout=0.05, train=False)
    targs = TrainingArguments(output_dir=str(WEIGHTS), model_max_length=5120,
                              fixed_mem_size=128, mean_compression_rate=4, bf16=True)
    lcfg = LoraConfig(r=512, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM")
    model = ICAE(margs, targs, lcfg)
    model.load_state_dict(load_file(str(WEIGHTS)), strict=False)
    return model.to(DEVICE).eval()


def compress(model, text):
    ids = model.tokenizer(text, truncation=True, max_length=MAX_LEN, padding=False,
                          return_attention_mask=False)["input_ids"]
    ids = torch.LongTensor([ids]).to(DEVICE)
    slots = model._compress(ids)            # [m, hidden]
    return slots.squeeze(0) if slots.dim() == 3 else slots


def decode(model, mean_slots):
    pr = model.tokenizer(LABEL_PROMPT, truncation=False, padding=False,
                         return_attention_mask=False, add_special_tokens=False)["input_ids"]
    right_ids = torch.LongTensor([[model.ft_token_id] + pr + INST_RIGHT]).to(DEVICE)
    left = model.tokens_to_embeddings(INST_LEFT)
    right = model.tokens_to_embeddings(right_ids)
    mem = mean_slots.to(right).unsqueeze(0)
    out_emb = torch.cat((left, mem, right), dim=1)
    toks, pkv = [], None
    for _ in range(MAX_NEW):
        with model.icae.disable_adapter():
            o = model.icae(inputs_embeds=out_emb, past_key_values=pkv, use_cache=True)
        logit = o.logits[:, -1, :model.vocab_size - 1]
        pkv = o.past_key_values
        nxt = torch.argmax(logit, dim=-1)
        if nxt.item() == 2:
            break
        out_emb = model.icae.get_base_model().model.embed_tokens(nxt).unsqueeze(1).to(DEVICE)
        toks.append(nxt.item())
    return model.tokenizer.decode(toks).strip()


def clean(text):
    text = text.strip().strip('"').strip()
    text = re.split(r"(?<=[.!?])\s", text)[0]      # first sentence
    text = re.sub(r"^(the\s+)?(common\s+)?(research\s+)?topic\s+(of\s+this\s+document\s+)?is\s*:?\s*",
                  "", text, flags=re.I)
    return text[:70].strip().rstrip(".")


def main():
    model = load_model()
    tree = json.loads(NODES.read_text())

    def label_node(node):
        docs = node.get("docs", [])[:K]
        texts = [f"Title: {d.get('title','')}. Abstract: {d.get('abstract','')}".strip()
                 for d in docs if (d.get("title") or d.get("abstract"))]
        if not texts:
            return ""
        with torch.no_grad():
            slots = [compress(model, t) for t in texts]
            m = min(s.shape[0] for s in slots)
            mean_slots = torch.stack([s[:m] for s in slots], 0).mean(0)
            raw = decode(model, mean_slots)
        return clean(raw)

    def walk(node):
        node["gen"] = label_node(node)
        node.pop("docs", None)
        print(f"  [{node.get('ref',''):>12}] {node['gen']}", flush=True)
        for c in node.get("children", []):
            walk(c)

    walk(tree)
    OUT.write_text(json.dumps(tree, ensure_ascii=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
