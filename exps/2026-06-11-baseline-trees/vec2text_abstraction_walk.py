"""vec2text counterpart of abstraction_walk.py: embed a single Wikipedia lead in
GTR space, scale the embedding toward 0, and invert to text at each radius. SAME
alphas as the doc2lora abstraction walk. vec2text is an inverter (no prompt).

Run (GPU): CUDA_VISIBLE_DEVICES=0 ../../.venv-vec2text/bin/python vec2text_abstraction_walk.py
"""
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ENTITIES = json.loads((HERE / "wiki_leads.json").read_text())
ALPHAS = [1.15, 0.90, 0.70, 0.55, 0.42, 0.30, 0.20, 0.12, 0.06]
NUM_STEPS = 20
EMB_MAXLEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = HERE / "vec2text_abstraction_walk.json"


def embed(encoder, tokenizer, texts):
    from vec2text.models.model_utils import mean_pool
    inp = tokenizer(texts, return_tensors="pt", max_length=EMB_MAXLEN,
                    truncation=True, padding="max_length").to(DEVICE)
    with torch.no_grad():
        h = encoder(input_ids=inp.input_ids, attention_mask=inp.attention_mask).last_hidden_state
    return mean_pool(h, inp.attention_mask)


def clean(t):
    return " ".join(t.strip().strip('"').split())[:90]


def main():
    import vec2text
    from transformers import AutoModel, AutoTokenizer
    encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
    corrector = vec2text.load_pretrained_corrector("gtr-base")
    print("vec2text loaded\n", flush=True)
    out = {}
    for etype, seed in ENTITIES.items():
        vec = embed(encoder, tokenizer, [seed])     # [1, 768]
        base_norm = float(vec.norm())
        steps = []
        for a in ALPHAS:
            inv = vec2text.invert_embeddings(vec * a, corrector=corrector, num_steps=NUM_STEPS)[0]
            steps.append({"alpha": a, "norm": base_norm * a, "label": clean(inv)})
            print(f"  [{etype}] a={a:.2f} norm={base_norm * a:8.2f}  {clean(inv)!r}", flush=True)
        out[etype] = {"seed": seed, "base_norm": base_norm, "steps": steps}
        OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
        print(flush=True)
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
