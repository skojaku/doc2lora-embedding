"""Decode an ABSTRACT from the fused embedding at every barycentric cell, then a
FOLLOW-UP 5-keyword list (multi-turn, same conversation) for doc2lora + in-context.

Per cell:
  turn 1: generate an abstract (4-6 sentences) from the fused adapter / prompt
  turn 2: continuing the SAME chat, ask for exactly 5 keyword phrases
Both saved.  Metrics (RGB, copy-rate) are later computed from the ABSTRACT; the
keywords are kept for figure clarity only.

Corner source = keyphrases + lead abstract (SRC_KW=1, canonical).  Writes
results/absfollow_<set>.json with cells: {doc2lora:{abs,kw}, incontext:{abs,kw}}.

  DOC2LORA_CKPT=/data2/.../qwen_4b_d2l/checkpoint-20000/pytorch_model.bin \
  PYTHONPATH=$DOC_TO_LORA_SRC \
  SRC_KW=1 CUDA_VISIBLE_DEVICES=0 python decode_absfollow.py strat00
"""
import os, sys, json, hashlib
from pathlib import Path
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simplex_common as sc
from doc2lora_legacy import (load_model, extract_norm_lora_emb,
                             internalize_from_norm_lora_emb, mix_embeddings)


def corner_fingerprint(spec):
    """identity of the corner CONTENT (not just the set name) -- guards against
    silently reusing a stale cache when a set name gets resampled to a different
    paper pair."""
    payload = "|".join(spec["leads"][k] for k in spec["keys"])
    return hashlib.md5(payload.encode()).hexdigest()

CKPT = os.environ.get(
    "DOC2LORA_CKPT",
    str(Path(__file__).resolve().parents[2] / "data/agent_assets/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"),
)

SUF = os.environ.get("KWTAG", "")
SRC_KW = os.environ.get("SRC_KW", "1") == "1"        # default ON (canonical)
OUT = SUF
CELL_SAMPLE = int(os.environ.get("CELL_SAMPLE", "0"))  # >0: decode only this many cells (stats mode)
PAIR_EDGE = os.environ.get("PAIR_EDGE", "0") == "1"    # only the A-B edge (C weight = 0): 2-corner interpolation


def grid_cells(set_name):
    """all 91 cells, or a seeded subsample (anchors + random) when CELL_SAMPLE>0."""
    import random
    if PAIR_EDGE:
        R = sc.GRID_DEN
        return [(k, R - k, 0) for k in range(R + 1)]
    cells = list(sc.barycentric_grid())
    if CELL_SAMPLE <= 0 or CELL_SAMPLE >= len(cells):
        return cells
    R = sc.GRID_DEN
    anchors = [(R, 0, 0), (0, R, 0), (0, 0, R), (R // 3, R // 3, R - 2 * (R // 3)),
               (R // 2, R // 2, 0), (R // 2, 0, R // 2), (0, R // 2, R // 2)]
    anchors = [a for a in anchors if a in cells]
    rng = random.Random(int("".join(ch for ch in set_name if ch.isdigit()) or "0"))
    rest = [c for c in cells if c not in anchors]
    rng.shuffle(rest)
    return anchors + rest[:max(0, CELL_SAMPLE - len(anchors))]
ABS_MAXTOK = int(os.environ.get("ABS_MAXTOK", "260"))
KW_MAXTOK = int(os.environ.get("KW_MAXTOK", "80"))

# ABS_PROMPT_ID selects a paraphrase of the decode instruction; 0 is the prompt the
# manuscript reports, so the default run is byte-identical to the original.
import psens_prompts as pp
ABS_PROMPT_ID = int(os.environ.get("ABS_PROMPT_ID", "0"))
SKIP_KW = os.environ.get("SKIP_KW", "0") == "1"   # prompt-sensitivity runs score the abstract only

ABS = pp.abs_prompt(ABS_PROMPT_ID)
KWQ = ("Now list exactly five short keyword phrases (two to four words each) "
       "naming the key concepts of that abstract. Output ONLY a comma-separated "
       "list, no sentences, no numbering.")
INC_ABS = pp.INC_PREFIX3 + ABS
INC_ABS2 = pp.INC_PREFIX2 + ABS


def corner_src(words, lead):
    if not SRC_KW:
        return lead
    kw = ", ".join(words) if isinstance(words, list) else str(words)
    return f"Keywords: {kw}.\n\n{lead}"


def gen(model, gen_tok, turns, max_new):
    ids = gen_tok.apply_chat_template(turns, tokenize=True, add_generation_prompt=True,
                                      return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                         max_new_tokens=max_new, do_sample=False)
    return gen_tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def abs_then_kw(model, gen_tok, first_user):
    """two-turn: abstract then keywords, keywords conditioned on the abstract."""
    turns = [{"role": "user", "content": first_user}]
    ab = gen(model, gen_tok, turns, ABS_MAXTOK)
    if SKIP_KW:
        return ab, ""
    turns += [{"role": "assistant", "content": ab}, {"role": "user", "content": KWQ}]
    kw = gen(model, gen_tok, turns, KW_MAXTOK)
    return ab, kw


def run_set(model, gen_tok, set_name):
    spec = sc.load_corners(sc.corners_path(set_name))
    keys = spec["keys"]
    V = {k: corner_src(spec["words"][k], spec["leads"][k]) for k in keys}
    Vraw = spec["leads"]                     # raw leads for the in-context prompt body
    out_path = os.path.join(sc.RESULTS, f"absfollow_{set_name}{OUT}.json")
    fp = corner_fingerprint(spec)
    done = {}
    if os.path.exists(out_path):
        existing = json.load(open(out_path))
        if existing.get("corner_fp") == fp:
            for c in existing["cells"]:
                done[tuple(c["bary"])] = c
        else:
            print(f"[{set_name}{OUT}] corner content changed -> ignoring stale cache", flush=True)

    genes = {}
    for k in keys:
        genes[k] = extract_norm_lora_emb(model, ctx_tok, V[k], max_length=512)
        model.reset()

    pair2 = len(keys) == 2   # genuine 2-corner spec (no third corner at all, not just weight-0)
    todo = grid_cells(set_name)
    cells = []
    for (i, j, k) in todo:
        bary = (i, j, k)
        if bary in done and done[bary].get("doc2lora", {}).get("abs") and done[bary].get("incontext", {}).get("abs"):
            cells.append(done[bary]); continue
        if pair2:
            w = [i / sc.GRID_DEN, j / sc.GRID_DEN]
            emb = mix_embeddings([genes[keys[0]], genes[keys[1]]], weights=w, mode="full")
            inc_first = INC_ABS2.format(A=Vraw[keys[0]], B=Vraw[keys[1]],
                                        pa=round(w[0]*100), pb=round(w[1]*100))
        else:
            w = [i / sc.GRID_DEN, j / sc.GRID_DEN, k / sc.GRID_DEN]
            emb = mix_embeddings([genes[keys[0]], genes[keys[1]], genes[keys[2]]],
                                 weights=w, mode="full")
            inc_first = INC_ABS.format(A=Vraw[keys[0]], B=Vraw[keys[1]], C=Vraw[keys[2]],
                                       pa=round(w[0]*100), pb=round(w[1]*100), pc=round(w[2]*100))
        internalize_from_norm_lora_emb(model, emb)
        d_ab, d_kw = abs_then_kw(model, gen_tok, ABS)
        model.reset()
        i_ab, i_kw = abs_then_kw(model, gen_tok, inc_first)
        model.reset()
        cells.append({"bary": [i, j, k], "weights": w,
                      "doc2lora": {"abs": d_ab, "kw": d_kw},
                      "incontext": {"abs": i_ab, "kw": i_kw}})
        if len(cells) % 5 == 0:
            json.dump({"set": set_name, "keys": keys, "corner_fp": fp, "cells": cells}, open(out_path, "w"))
            print(f"[{set_name}{OUT}] {len(cells)}/{len(todo)}", flush=True)
    json.dump({"set": set_name, "keys": keys, "corner_fp": fp, "cells": cells}, open(out_path, "w"))
    print(f"wrote {out_path} ({len(cells)} cells)")


def main(set_names):
    global ctx_tok  # module-level so run_set (a plain function, not a closure) can see it
    model, gen_tok, ctx_tok = load_model(CKPT)
    for s in set_names:
        run_set(model, gen_tok, s)


if __name__ == "__main__":
    main(sys.argv[1:])
