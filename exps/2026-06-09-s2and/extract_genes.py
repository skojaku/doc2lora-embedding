"""[GPU] Extract doc2lora gemma idea-genes for an S2AND sub-dataset's papers, via the lib API.
Uses libs/doc2lora load_model(mode="embed") (frees the generator half -> avoids OOM) and a capped
max token length. norm_lora_emb [26,1,8,512] -> mean over rank -> flatten 13312 (field-gene format).

Usage: CUDA_VISIBLE_DEVICES=0 HF_HOME=... HF_TOKEN=... python extract_genes.py zbmath
Output: proc/<ds>/genes_gemma.npz  (paper_ids[str], embeddings[N,13312] fp16)
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "libs/legacy"))
from doc2lora_legacy import load_model, extract_norm_lora_emb_batch

DS = sys.argv[1] if len(sys.argv) > 1 else "zbmath"
ENC = sys.argv[2] if len(sys.argv) > 2 else "gemma"
CKPTS = {"gemma": "data/agent_assets/gemma_demo/checkpoint-80000/pytorch_model.bin",
         "qwen": "data/agent_assets/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin",
         "mistral": "data/agent_assets/mistral_7b_d2l/checkpoint-20000/pytorch_model.bin"}
CKPT = os.environ.get("DOC2LORA_CKPT", CKPTS[ENC])
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
BATCH = int(os.environ.get("BATCH", "8"))
base = "exps/2026-06-09-s2and"

pt = pd.read_parquet(f"{base}/proc/{DS}/paper_text.parquet")
pids = pt.paper_id.astype(str).tolist()
texts = [t if isinstance(t, str) and t.strip() else "." for t in pt.text.tolist()]
print(f"[{DS}/{ENC}] {len(texts)} papers | max_tokens={MAX_TOKENS} batch={BATCH}", flush=True)

model, tok, ctx = load_model(CKPT, mode="embed")     # frees generator half
out = None
t0 = time.time(); CH = 256
for i in range(0, len(texts), CH):
    embs = extract_norm_lora_emb_batch(model, ctx, texts[i:i + CH], max_length=MAX_TOKENS, batch_size=BATCH)
    for j, e in enumerate(embs):
        v = e.float().mean(dim=2).reshape(-1).cpu().to(torch.float16).numpy()   # L*512, L = model layers
        if out is None:
            out = np.empty((len(texts), v.shape[0]), np.float16)
        out[i + j] = v
    del embs; torch.cuda.empty_cache()
    print(f"  {min(i+CH,len(texts))}/{len(texts)}  [{time.time()-t0:.0f}s]", flush=True)

op = f"{base}/proc/{DS}/genes_{ENC}.npz"
np.savez(op, paper_ids=np.array(pids), embeddings=out)
print(f"[saved] {op}  dim={out.shape[1]} ({time.time()-t0:.0f}s)", flush=True)
