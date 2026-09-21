"""[GPU] Invertibility demo (ICLR): the per-token ICAE adapter is exactly invertible, so adapted
memory slots can be mapped back to the original slots and DECODED to text through the frozen ICAE
decoder. Confirms the "task-competitive AND invertible/decodable" property for ICAE, paralleling
doc2lora. Reports reconstruction cos and a decoded text sample.

Usage: CUDA_VISIBLE_DEVICES=0 python decode_roundtrip.py [n=3]
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "exps/2026-06-09-kron-adapter")
from icae_lib import load_icae, compress_batch, MAX_LEN
from kron import KronAdapter

W = "exps/2026-05-26-baseline/icae/weights/mistral_7b_ft_icae.safetensors"
CODE = "exps/2026-05-26-baseline/icae/code/icae_v2"
BASE = "mistralai/Mistral-7B-Instruct-v0.2"
dev = "cuda"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
INST_LEFT = torch.LongTensor([[1, 733, 16289, 28793]])
INST_RIGHT = [733, 28748, 16289, 28793]
PROMPT = "Briefly restate what this document is about."


@torch.no_grad()
def decode(model, slots, max_new=64):
    pr = model.tokenizer(PROMPT, add_special_tokens=False)["input_ids"]
    right = torch.LongTensor([[model.ft_token_id] + pr + INST_RIGHT]).to(dev)
    le = model.tokens_to_embeddings(INST_LEFT.to(dev)); ri = model.tokens_to_embeddings(right)
    emb = torch.cat((le, slots.to(ri).unsqueeze(0), ri), dim=1)
    toks, pkv = [], None
    for _ in range(max_new):
        with model.icae.disable_adapter():
            o = model.icae(inputs_embeds=emb, past_key_values=pkv, use_cache=True)
        pkv = o.past_key_values
        nxt = torch.argmax(o.logits[:, -1, :model.vocab_size - 1], dim=-1)
        if nxt.item() == 2:
            break
        emb = model.icae.get_base_model().model.embed_tokens(nxt).unsqueeze(1).to(dev)
        toks.append(nxt.item())
    return model.tokenizer.decode(toks).strip()


df = pd.read_parquet("exps/2026-06-10-general-adapter/pool_text.parquet").head(N)
texts = [t if isinstance(t, str) and t.strip() else "." for t in df.text.tolist()]
model = load_icae(W, CODE, BASE, dev)
ck = torch.load(f"{HERE}/adapter_icae.pt", map_location=dev)
A = KronAdapter(L=ck["L"], d=ck["D"]).to(dev); A.load_state_dict(ck["state"]); A.eval()

with torch.no_grad():
    S = compress_batch(model, texts, dev)                      # [N,128,4096]
    C = A.C(); alpha = torch.exp(A.alpha)
    Z = torch.einsum("bld,de->ble", S, C) * alpha[None, :, None]          # adapted slots
    Cinv = torch.linalg.inv(C)
    Srec = torch.einsum("ble,ed->bld", Z / alpha[None, :, None], Cinv)    # invert -> original slots
    for i, t in enumerate(texts):
        cos = F.cosine_similarity(Srec[i].flatten(), S[i].flatten(), dim=0).item()
        print(f"\n--- doc{i}  recon cos={cos:.5f}")
        print(f"  orig  : {t[:140]}")
        print(f"  decode(original slots) : {decode(model, S[i])[:160]}")
        print(f"  decode(inv(adapted))   : {decode(model, Srec[i])[:160]}")
