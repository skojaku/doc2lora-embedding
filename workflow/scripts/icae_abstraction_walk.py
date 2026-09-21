"""ICAE counterpart of abstraction_walk.py: compress a single Wikipedia lead,
scale its memory slots toward 0, and decode at each radius. Uses the SAME prompt
and SAME alphas as the doc2lora abstraction walk for a like-for-like comparison.

Run (GPU): CUDA_VISIBLE_DEVICES=0 ~/miniforge3/envs/doc2lora/bin/python icae_abstraction_walk.py
"""
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
from bench_data import out_dir   # bench_data.py sits next to this file
DATA = out_dir("labels")           # where this chain writes


# doc2lora abstraction_walk.py prompt (override ICAE's default field-naming prompt
# so both methods receive an identical instruction).
PROMPT = ("In two to four words, name the topic or category that best describes "
          "this document. Reply with only the name, nothing else.")
import icae_label  # noqa: E402
icae_label.LABEL_PROMPT = PROMPT
from icae_label import load_model, compress, decode, clean  # noqa: E402

ENTITIES = json.loads((HERE / "wiki_leads.json").read_text())
ALPHAS = [1.15, 0.90, 0.70, 0.55, 0.42, 0.30, 0.20, 0.12, 0.06]
OUT = DATA / "icae_abstraction_walk.json"


def main():
    model = load_model()
    print(f"ICAE loaded; prompt={PROMPT!r}\n", flush=True)
    out = {}
    for etype, seed in ENTITIES.items():
        with torch.no_grad():
            slots = compress(model, seed)          # [m, hidden]
        base_norm = float(slots.norm())
        steps = []
        for a in ALPHAS:
            with torch.no_grad():
                lab = clean(decode(model, slots * a))
            steps.append({"alpha": a, "norm": base_norm * a, "label": lab})
            print(f"  [{etype}] a={a:.2f} norm={base_norm * a:8.2f}  {lab!r}", flush=True)
        out[etype] = {"seed": seed, "base_norm": base_norm, "steps": steps}
        OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
        print(flush=True)
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
