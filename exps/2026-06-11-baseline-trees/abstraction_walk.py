"""Walk a specific entity's gene toward the origin and decode at each radius.

Tests whether ||v||_2 (radius) tracks abstraction: a specific entity decoded at
full magnitude, then scaled toward 0 (smaller radius), should yield progressively
more general/abstract labels. Run across diverse entity types to show universality.

Output: abstraction_walk.json  {type: {"seed": text, "steps":[{alpha,norm,label}]}}

Run (GPU): CUDA_VISIBLE_DEVICES=0 python abstraction_walk.py
"""
import json
import re
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]   # repository root, not a fixed ~/projects path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "libs/legacy"))   # frozen doc2lora_legacy API
from doc2lora_legacy import (  # noqa: E402
    load_model, extract_norm_lora_emb, internalize_from_norm_lora_emb, generate_text,
)

CKPT = Path(os.environ.get(
    "DOC2LORA_CKPT",
    ROOT / "data/agent_assets/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"))
OUT = HERE / "abstraction_walk.json"
ALPHAS = [1.15, 0.90, 0.70, 0.55, 0.42, 0.30, 0.20, 0.12, 0.06]
PROMPT = ("In two to four words, name the topic or category that best describes "
          "this document. Reply with only the name, nothing else.")

# starting descriptions = Wikipedia lead paragraphs (wiki_leads.json)
ENTITIES = json.loads((HERE / "wiki_leads.json").read_text())


def normalize_label(text):
    lines = [l.strip(" \t\"'*`-") for l in text.splitlines() if l.strip()]
    lab = lines[0] if lines else text.strip()
    lab = re.sub(r"^(topic|category|label|field)\s*[:\-]\s*", "", lab, flags=re.I)
    return lab.strip(" \t\"'*`.")


def main():
    model, gen_tok, ctx_tok = load_model(str(CKPT), mode="full")
    out = {}
    for etype, seed in ENTITIES.items():
        gene = extract_norm_lora_emb(model, ctx_tok, seed).float().cpu()  # [36,1,8,512]
        base_norm = float(gene.norm())
        steps = []
        for a in ALPHAS:
            t = (gene * a).contiguous().to(model.device)
            model.reset()
            internalize_from_norm_lora_emb(model, t)
            label = normalize_label(
                generate_text(model, gen_tok, PROMPT, max_new_tokens=16))
            model.reset()
            steps.append({"alpha": a, "norm": base_norm * a, "label": label})
            print(f"  [{etype}] a={a:.2f} norm={base_norm*a:7.2f}  {label}", flush=True)
        out[etype] = {"seed": seed, "base_norm": base_norm, "steps": steps}
        OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
        print()
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
