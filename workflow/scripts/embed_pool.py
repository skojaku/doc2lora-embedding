"""[GPU] Fresh doc2lora embedding of the sampled OpenAlex pool (pool_text.parquet) for one encoder.
norm_lora_emb [L,1,8,512] -> mean over rank -> flatten (L*512). Lib API, mode=embed, capped tokens.
Usage: CUDA_VISIBLE_DEVICES=0 python embed_pool.py gemma
Output: pool_genes_<enc>.npz  (gkey[str], embeddings[N, L*512] fp16)
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "libs/legacy"))
from doc2lora_legacy import load_model, extract_norm_lora_emb_batch

ENC = sys.argv[1] if len(sys.argv) > 1 else "gemma"
CKPTS = {"gemma": "data/agent_assets/gemma_demo/checkpoint-80000/pytorch_model.bin",
         "qwen": "data/agent_assets/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin",
         "mistral": "data/agent_assets/mistral_7b_d2l/checkpoint-20000/pytorch_model.bin"}
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
BATCH = int(os.environ.get("BATCH", "8"))
OUT = "data/general_adapter"

df = pd.read_parquet(os.environ.get("POOL_TEXT", f"{OUT}/pool_text.parquet"))
keys = df.pid.tolist()
texts = [t if isinstance(t, str) and t.strip() else "." for t in df.text.tolist()]
print(f"[{ENC}] embedding {len(texts):,} pool papers (max_tokens={MAX_TOKENS} batch={BATCH})", flush=True)

model, tok, ctx = load_model(CKPTS[ENC], mode="embed")
out = None; t0 = time.time(); CH = 256
for i in range(0, len(texts), CH):
    embs = extract_norm_lora_emb_batch(model, ctx, texts[i:i + CH], max_length=MAX_TOKENS, batch_size=BATCH)
    for j, e in enumerate(embs):
        v = e.float().mean(dim=2).reshape(-1).cpu().to(torch.float16).numpy()
        if out is None:
            out = np.empty((len(texts), v.shape[0]), np.float16)
        out[i + j] = v
    del embs; torch.cuda.empty_cache()
    if i % 2560 == 0:
        print(f"  {min(i+CH,len(texts))}/{len(texts)} [{time.time()-t0:.0f}s]", flush=True)

_out = os.environ.get("POOL_OUT", f"{OUT}/pool_genes_{ENC}.npz")
np.savez(_out, pids=np.array(keys), embeddings=out)
print(f"[saved] {_out} dim={out.shape[1]} ({time.time()-t0:.0f}s)", flush=True)
