"""ICAE version of decode_absfollow.py: abstract from mixed memory slots, then a
follow-up 5-keyword list conditioned on slots + the generated abstract.

  PYTHONPATH=$DOC_TO_LORA_SRC \
  SRC_KW=1 CUDA_VISIBLE_DEVICES=0 python decode_absfollow_icae.py strat00
"""
import os, sys, json, hashlib
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simplex_common as sc
import icae_slots as ici


def corner_fingerprint(spec):
    """identity of the corner CONTENT (not just the set name) -- guards against
    silently reusing a stale cache when a set name gets resampled to a different
    paper pair."""
    payload = "|".join(spec["leads"][k] for k in spec["keys"])
    return hashlib.md5(payload.encode()).hexdigest()


def decode_sentence(model, slots, prompt, max_new=190):
    pr = model.tokenizer(prompt, truncation=False, padding=False,
                         return_attention_mask=False, add_special_tokens=False)["input_ids"]
    right_ids = torch.LongTensor([[model.ft_token_id] + pr + ici.INST_RIGHT]).to(ici.DEVICE)
    left = model.tokens_to_embeddings(ici.INST_LEFT)
    right = model.tokens_to_embeddings(right_ids)
    mem = slots.to(right).unsqueeze(0)
    out_emb = torch.cat((left, mem, right), dim=1)
    toks, pkv = [], None
    for _ in range(max_new):
        with model.icae.disable_adapter():
            o = model.icae(inputs_embeds=out_emb, past_key_values=pkv, use_cache=True)
        logit = o.logits[:, -1, :model.vocab_size - 1]
        pkv = o.past_key_values
        nxt = torch.argmax(logit, dim=-1)
        if nxt.item() == 2:                         # </s>
            break
        out_emb = model.icae.get_base_model().model.embed_tokens(nxt).unsqueeze(1).to(ici.DEVICE)
        toks.append(nxt.item())
    return model.tokenizer.decode(toks).strip()

SUF = os.environ.get("KWTAG", "")
SRC_KW = os.environ.get("SRC_KW", "1") == "1"
OUT = SUF
CELL_SAMPLE = int(os.environ.get("CELL_SAMPLE", "0"))
PAIR_EDGE = os.environ.get("PAIR_EDGE", "0") == "1"    # only the A-B edge (C weight = 0): 2-corner interpolation


def grid_cells(set_name):
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
# ABS_PROMPT_ID selects a paraphrase of the decode instruction; 0 is the prompt the
# manuscript reports, so the default run is byte-identical to the original.
import psens_prompts as pp
ABS_PROMPT_ID = int(os.environ.get("ABS_PROMPT_ID", "0"))
SKIP_KW = os.environ.get("SKIP_KW", "0") == "1"   # prompt-sensitivity runs score the abstract only

ABS = pp.abs_prompt(ABS_PROMPT_ID)
KWQ = ("List exactly five short keyword phrases (two to four words each) naming "
       "the key concepts. Output ONLY a comma-separated list.")


def corner_src(words, lead):
    if not SRC_KW:
        return lead
    kw = ", ".join(words) if isinstance(words, list) else str(words)
    return f"Keywords: {kw}.\n\n{lead}"


def run_set(model, set_name):
    spec = sc.load_corners(sc.corners_path(set_name))
    keys = spec["keys"]
    out_path = os.path.join(sc.RESULTS, f"absfollow_{set_name}{OUT}_icae.json")
    fp = corner_fingerprint(spec)
    done = {}
    if os.path.exists(out_path):
        existing = json.load(open(out_path))
        if existing.get("corner_fp") == fp:
            for c in existing["cells"]:
                done[tuple(c["bary"])] = c
        else:
            print(f"[{set_name}{OUT} icae] corner content changed -> ignoring stale cache", flush=True)

    pair2 = len(keys) == 2   # genuine 2-corner spec

    with torch.no_grad():
        slots = {k: ici.compress(model, corner_src(spec["words"][k], spec["leads"][k])) for k in keys}
    m = min(s.shape[0] for s in slots.values())
    S = {k: slots[k][:m] for k in keys}

    todo = grid_cells(set_name)
    cells = []
    for (i, j, k) in todo:
        bary = (i, j, k)
        if bary in done and done[bary].get("icae", {}).get("abs"):
            cells.append(done[bary]); continue
        if pair2:
            w = [i / sc.GRID_DEN, j / sc.GRID_DEN]
            mixed = w[0] * S[keys[0]] + w[1] * S[keys[1]]
        else:
            w = [i / sc.GRID_DEN, j / sc.GRID_DEN, k / sc.GRID_DEN]
            mixed = w[0] * S[keys[0]] + w[1] * S[keys[1]] + w[2] * S[keys[2]]
        with torch.no_grad():
            ab = decode_sentence(model, mixed, ABS, max_new=260)
            # follow-up: keywords conditioned on slots + the abstract just written
            kw = "" if SKIP_KW else decode_sentence(model, mixed, f"{ab}\n\n{KWQ}", max_new=80)
        cells.append({"bary": [i, j, k], "weights": w, "icae": {"abs": ab, "kw": kw}})
        if len(cells) % 5 == 0:
            json.dump({"set": set_name, "keys": keys, "corner_fp": fp, "cells": cells}, open(out_path, "w"))
            print(f"[{set_name}{OUT} icae] {len(cells)}/{len(todo)}", flush=True)
    json.dump({"set": set_name, "keys": keys, "corner_fp": fp, "cells": cells}, open(out_path, "w"))
    print(f"wrote {out_path} ({len(cells)} cells)")


def main(set_names):
    model = ici.load_model()
    for s in set_names:
        run_set(model, s)


if __name__ == "__main__":
    main(sys.argv[1:])
