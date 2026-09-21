"""[GPU] Encoder cost benchmark for #71: throughput, peak VRAM and parameter count.

One method per process so peak VRAM is attributable.  All methods see the SAME documents in the same
order on the same card, and every measurement excludes model load and a warm-up pass.

methods: d2l-qwen | d2l-mistral | d2l-gemma | icae | sbert | specter2

Usage: NEED_MB=24000 bash workflow/scripts/gpu_lease.sh \
         python workflow/scripts/groupc/eff_throughput.py --method sbert --out ...
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d2l  # noqa: E402

CKPTS = {
    "d2l-qwen": "data/agent_assets/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin",
    "d2l-mistral": "data/agent_assets/mistral_7b_d2l/checkpoint-20000/pytorch_model.bin",
    "d2l-gemma": "data/agent_assets/gemma_demo/checkpoint-80000/pytorch_model.bin",
}
HF = {"sbert": "sentence-transformers/all-mpnet-base-v2", "specter2": "allenai/specter2_base"}


def n_params(m) -> int:
    try:
        return int(sum(p.numel() for p in m.parameters()))
    except Exception:  # noqa: BLE001
        return 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--sample", default="data/groupc/fidelity/sample.parquet")
    p.add_argument("--out", required=True)
    p.add_argument("--n_docs", type=int, default=256)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max_batch_tokens", type=int, default=6144)
    a = p.parse_args()

    texts = pd.read_parquet(a.sample).text.astype(str).tolist()
    docs = (texts * (a.n_docs // max(len(texts), 1) + 1))[:a.n_docs]
    warm = docs[:a.warmup]
    dev = "cuda"
    torch.cuda.reset_peak_memory_stats()
    rec = {"method": a.method, "n_docs": len(docs)}

    if a.method.startswith("d2l"):
        model, gen_tok, ctx_tok = d2l.load(CKPTS[a.method], mode="embed")
        rec["params"] = n_params(model)
        d2l.extract_full(model, ctx_tok, warm, max_batch_tokens=a.max_batch_tokens)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        out = []
        for c0 in range(0, len(docs), 64):
            out += d2l.extract_full(model, ctx_tok, docs[c0:c0 + 64],
                                    max_batch_tokens=a.max_batch_tokens)
        torch.cuda.synchronize()
        el = time.time() - t0
        e0 = out[0]
        full = int(np.prod(tuple(e0.shape)))
        rec.update(dim_full=full, dim_pooled=full // e0.shape[-2] if e0.dim() >= 2 else full)

    elif a.method == "icae":
        sys.path.insert(0, "exps/2026-06-21-icae-benchmark")
        from icae_lib import compress_batch, load_icae
        model = load_icae("exps/2026-05-26-baseline/icae/weights/mistral_7b_ft_icae.safetensors",
                          "exps/2026-05-26-baseline/icae/code/icae_v2",
                          "mistralai/Mistral-7B-Instruct-v0.2", dev)
        rec["params"] = n_params(model)
        with torch.no_grad():
            compress_batch(model, warm, dev)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            for c0 in range(0, len(docs), 24):
                compress_batch(model, docs[c0:c0 + 24], dev)
            torch.cuda.synchronize()
            el = time.time() - t0
        rec.update(dim_full=128 * 4096, dim_pooled=4096)

    else:
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModel, AutoTokenizer
        if a.method == "sbert":
            m = SentenceTransformer(HF["sbert"], device=dev)
            rec["params"] = n_params(m)
            m.encode(warm, batch_size=a.batch, show_progress_bar=False)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            V = m.encode(docs, batch_size=a.batch, convert_to_numpy=True, show_progress_bar=False)
            torch.cuda.synchronize(); el = time.time() - t0
            rec.update(dim_full=V.shape[1], dim_pooled=V.shape[1])
        else:
            tok = AutoTokenizer.from_pretrained(HF["specter2"])
            m = AutoModel.from_pretrained(HF["specter2"]).to(dev).eval()
            rec["params"] = n_params(m)

            def run(batch):
                enc = tok(batch, padding=True, truncation=True, max_length=512,
                          return_tensors="pt").to(dev)
                with torch.no_grad():
                    return m(**enc).last_hidden_state[:, 0, :]
            run(warm)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            d = 0
            for c0 in range(0, len(docs), a.batch):
                d = run(docs[c0:c0 + a.batch]).shape[1]
            torch.cuda.synchronize(); el = time.time() - t0
            rec.update(dim_full=d, dim_pooled=d)

    rec.update(seconds=el, docs_per_sec=len(docs) / el,
               peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    main()
