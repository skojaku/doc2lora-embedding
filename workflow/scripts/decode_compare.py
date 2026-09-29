"""Generate decodes for one arm of the gene-vs-activation comparison (#152).

Generation and scoring are separate on purpose: `score_compare.py` reads the decode
files, so the degeneracy guard and the distractor protocol can be revised without
re-running the GPU work.

Arms (`--arm`):
  `act`   Qwen3-4B layer-15 hidden states, mean-pooled, patched into 8 slots of the
          continuation prompt. That configuration is the winner of `layer_sweep.py`
          (MRR 0.284 vs a chance of 0.090), not a hand-picked setting.
  `gene`  the \\doctolora full-rank `norm_lora_emb`, injected as a LoRA and decoded
          through the same Qwen3-4B.

Both arms decode the SAME documents with the SAME generation budget. The prompts
cannot be identical -- patching needs a placeholder position and a gene does not -- and
that asymmetry is recorded in NOTE.md rather than papered over.

The `floor` arm (nothing injected) is written too: the continuation prompt has a strong
prior toward biomedical abstracts, so every score is a lift over it, never an absolute.

Documents carry their PACS subdivision so `score_compare.py` can draw same-subfield
distractors. Random distractors previously manufactured a win on next-paper prediction
that vanished under same-subfield ones, so the hard protocol is the default here.

  CUDA_VISIBLE_DEVICES=1 python decode_compare.py --arm act  --n 500
  CUDA_VISIBLE_DEVICES=0 python decode_compare.py --arm gene --n 500
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
from bench_data import out_dir   # bench_data.py sits next to this file
DATA = out_dir("actpatch")           # where this chain writes

ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

BEST = dict(layer=15, slots=8, mode="mean")     # from results/layer_sweep.json
CA = ROOT / "data/pacs"
MAX_NEW = 64


def sample_docs(n, seed=5, min_per_sub=120):
    """APS papers with a PACS subdivision, from subdivisions big enough to supply
    99 same-subfield distractors later."""
    pg = pd.read_parquet(CA / "results/paper_groups.parquet",
                         columns=["paper_id", "subdivision"])
    pg = pg[pg.subdivision.notna()]
    pg["subdivision"] = pg.subdivision.astype(str)
    big = pg.subdivision.value_counts()
    pg = pg[pg.subdivision.isin(big[big >= min_per_sub].index)]

    txt = pd.read_parquet(ROOT / "data/aps/paper_text_pid.parquet",
                          columns=["paper_id", "title", "abstract"])
    txt = txt[txt.abstract.notna() & (txt.abstract.str.len() > 200)]
    txt = txt.assign(title=txt.title.fillna("").str.strip())
    df = txt.merge(pg, on="paper_id", how="inner")
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    df = df.assign(text=(df.title + ". " + df.abstract.str.strip()).str.strip(". "))
    return df


# --------------------------------------------------------------------------- #
def run_act(df, batch_size=16):
    from act_common import load_qwen, extract_acts
    from layer_sweep import decode_batch, floor_decode

    Q = load_qwen()
    V = extract_acts(Q, list(df.text), BEST["layer"], mode=BEST["mode"])
    dec = decode_batch(Q, V, BEST["layer"], BEST["slots"],
                       max_new_tokens=MAX_NEW, batch_size=batch_size)
    return dec, floor_decode(Q, BEST["slots"], max_new_tokens=MAX_NEW)


def run_gene(df):
    """doc2lora full-rank gene -> LoRA -> decode, through the same Qwen3-4B."""
    sys.path.insert(0, str(ROOT / "libs/doc2lora"))
    from doc2lora import load_model, extract_norm_lora_emb, decode_adapter

    ckpt = os.environ.get(
        "DOC2LORA_CKPT",
        str(ROOT / "data/agent_assets/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"))
    model, gen_tok, ctx_tok = load_model(ckpt, mode="full")
    # the same instruction the activation arm continues from, phrased for a gene
    prompt = "The following is a scientific abstract. Write it."
    out = []
    for i, t in enumerate(df.text):
        emb = extract_norm_lora_emb(model, ctx_tok, t)
        out.append(decode_adapter(model, gen_tok, emb, prompt=prompt,
                                  max_new_tokens=MAX_NEW))
        if i % 25 == 0:
            print(f"  [{i}/{len(df)}] {out[-1][:90]}", flush=True)
    floor = decode_adapter(model, gen_tok, torch.zeros_like(emb), prompt=prompt,
                           max_new_tokens=MAX_NEW)
    return out, floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["act", "gene"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_path = Path(a.out or DATA / f"decodes_{a.arm}.json")

    df = sample_docs(a.n)
    print(f"[data] {len(df)} papers across {df.subdivision.nunique()} PACS subdivisions",
          flush=True)

    if a.arm == "act":
        dec, floor = run_act(df, batch_size=a.batch_size)
    else:
        dec, floor = run_gene(df)

    rec = dict(
        arm=a.arm, n=len(df), config=BEST if a.arm == "act" else {"full_rank": True},
        floor=floor,
        items=[{"paper_id": int(p), "subdivision": s, "title": t,
                "abstract": ab, "decode": d}
               for p, s, t, ab, d in zip(df.paper_id, df.subdivision, df.title,
                                         df.abstract, dec)],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec, indent=2))
    print(f"[floor] {floor[:140]}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
