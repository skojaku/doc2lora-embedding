"""Decode the FULL-RANK node means with the field (2-3 word) prompt -- VERBATIM.

This is the exact in-model Doc2LoRA prompt used to produce qwen_fullrank_field23.json
(the labels shown in the figures/tables). Recorded here so the prompt is no longer
reconstructed. Internalizes each node's full-rank [36,8,512] mean (8 distinct rank
slots, no broadcast, no renorm) and decodes greedily.

Input:  qwen_fullrank_means.npz
Output: qwen_fullrank_field23.json   {node: field-label}

Run (GPU): CUDA_VISIBLE_DEVICES=0 python decode_fullrank_field23.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]   # repository root, not a fixed ~/projects path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "libs/doc2lora"))
from doc2lora_legacy import (  # noqa: E402
    load_model, internalize_from_norm_lora_emb, generate_text,
)

CKPT = ROOT / "data/agent_assets/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"
MEANS = HERE / "qwen_fullrank_means.npz"
OUT = HERE / "qwen_fullrank_field23.json"
L, R, LAT = 36, 8, 512

# VERBATIM field (2-3 word) prompt used for the Doc2LoRA labels in the figures/tables.
FIELD23_PROMPT = (
    "In 2 to 3 words, name the scientific field that all of these documents "
    "belong to. Reply with only the field name."
)


def normalize_label(text):
    lines = [l.strip(" \t\"'*`-") for l in text.splitlines() if l.strip()]
    lab = lines[0] if lines else text.strip()
    lab = re.sub(r"^(cluster label|cluster|label|topic|field)\s*[:\-]\s*", "",
                 lab, flags=re.I)
    return lab.strip(" \t\"'*`.")


def main():
    z = np.load(MEANS, allow_pickle=True)
    means = z["means"].astype(np.float32)
    nodes = z["nodes"].astype(str)

    model, gen_tok, ctx_tok = load_model(str(CKPT), mode="full")
    out = {}
    for key, m in zip(nodes, means):
        t = torch.from_numpy(m).reshape(L, 1, R, LAT).contiguous().to(model.device)
        model.reset()
        internalize_from_norm_lora_emb(model, t)
        out[key] = normalize_label(
            generate_text(model, gen_tok, FIELD23_PROMPT, max_new_tokens=16))
        model.reset()
        print(f"  [{key}] {out[key]}", flush=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
