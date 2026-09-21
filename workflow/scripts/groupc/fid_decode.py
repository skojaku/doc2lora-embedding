"""[GPU] Decode single-document points for the #95 fidelity test, one condition per run.

The manuscript's evidence that a Doc2LoRA point is faithful to the document it came from is one
cacio e pepe recipe on Mistral-7B, while every quantitative result uses Qwen3-4B.  This produces the
missing measurement on real papers with the encoder the paper actually reports, against a ceiling
and two floors:

  d2l        the paper's own gene is internalised; the prompt contains NO paper text
  incontext  no adapter, the paper text IS in the prompt (ceiling: the model reading the paper)
  mismatch   a DIFFERENT paper's gene is internalised, same prompt (floor: prompt + base-model prior
             dressed in a real adapter)
  prior      no adapter, no text (absolute floor: what the question alone elicits)

Same base model, same greedy decoding and same question in all four, so the only thing that varies
is what the model has access to.

Usage: NEED_MB=20000 bash workflow/scripts/gpu_lease.sh \
         python workflow/scripts/groupc/fid_decode.py --cond d2l --sample ... --out ...
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d2l  # noqa: E402

# One fixed question for every condition (greedy, no sampling).
QUESTION = (
    "Describe the scientific paper you have read. State the topic, the system or object studied, "
    "the method used, and the main finding. Answer in at most four sentences."
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cond", choices=["d2l", "incontext", "mismatch", "prior"], required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--max_new_tokens", type=int, default=160)
    p.add_argument("--max_ctx_chars", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    df = pd.read_parquet(a.sample)
    pids = df.paper_id.astype(np.int64).tolist()
    texts = df.text.astype(str).tolist()
    print(f"[fid-decode {a.cond}] {len(pids)} papers", flush=True)

    mode = "generate" if a.cond in ("incontext", "prior") else "full"
    model, gen_tok, ctx_tok = d2l.load(a.ckpt, mode=mode)

    # mismatch: a fixed derangement of the sample (every paper gets some other paper's adapter)
    rng = np.random.default_rng(a.seed)
    perm = np.arange(len(pids))
    if a.cond == "mismatch":
        while True:
            perm = rng.permutation(len(pids))
            if not (perm == np.arange(len(pids))).any():
                break

    out = {}
    for i, (pid, txt) in enumerate(zip(pids, texts)):
        if a.cond in ("d2l", "mismatch"):
            src = perm[i] if a.cond == "mismatch" else i
            emb = d2l.extract_full(model, ctx_tok, [texts[src]])[0]
            raw = d2l.decode(model, gen_tok, emb.detach().cpu().numpy(), QUESTION,
                             max_new_tokens=a.max_new_tokens)
            rec = {"decode": raw.strip(), "source_row": int(src), "source_pid": int(pids[src])}
        elif a.cond == "incontext":
            prompt = f"Paper:\n{txt[:a.max_ctx_chars]}\n\n{QUESTION}"
            model.reset()
            from doc2lora_legacy import generate_text
            raw = generate_text(model, gen_tok, prompt, max_new_tokens=a.max_new_tokens)
            rec = {"decode": raw.strip()}
        else:  # prior -- greedy and input-free, so it is the same string for every row
            if not out:
                model.reset()
                from doc2lora_legacy import generate_text
                _prior = generate_text(model, gen_tok, QUESTION, max_new_tokens=a.max_new_tokens)
            rec = {"decode": _prior.strip()}
        out[str(pid)] = rec
        if i % 20 == 0:
            print(f"  {i}/{len(pids)}  [{pid}] {rec['decode'][:90]!r}", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"condition": a.cond, "question": QUESTION, "decodes": out}, fh,
                  ensure_ascii=False, indent=1)
    print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
