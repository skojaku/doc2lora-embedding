"""Ask each model (doc2lora, in-context, icae) DIRECTLY for its own one-sentence,
plain-English, non-expert summary -- continuing the SAME decode conversation
that produced the cached abstract+keywords, at the 3 quarter cells (25/50/75%)
of a pair-axis set. Writes/updates keyword_summaries.json, which
pair_axis_colorband.py already reads as an override for the keyword-list labels.

  PYTHONPATH=$DOC_TO_LORA_SRC \
  DOC2LORA_CKPT=data/agent_assets/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin \
  CUDA_VISIBLE_DEVICES=0 python summarize_quarters.py pairCSML_00
"""
import os, sys, json
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simplex_common as sc
from doc2lora_legacy import (load_model, extract_norm_lora_emb,
                             internalize_from_norm_lora_emb, mix_embeddings)

ARGV = [a for a in sys.argv[1:] if not a.startswith("--")]
ONLY = os.environ.get("ONLY", "")   # "icae" to skip the qwen (doc2lora+incontext) half
SET = ARGV[0] if ARGV else "pairCSML_00"
TAG = "_pairaxis"
QUARTERS = [0.25, 0.5, 0.75]
R = sc.GRID_DEN
HERE = os.path.dirname(os.path.abspath(__file__))

ABS_PROMPT = ("Write a detailed abstract (four to six sentences) describing this "
             "research topic: its problem, methods, and findings.")
KWQ = ("Now list exactly five short keyword phrases (two to four words each) "
       "naming the key concepts of that abstract. Output ONLY a comma-separated "
       "list, no sentences, no numbering.")
INC_ABS2 = ("You are given two short documents:\n\n[A] {A}\n\n[B] {B}\n\n"
           "Imagine ONE research topic that blends these two in the proportions "
           "A:{pa}%, B:{pb}%. " + ABS_PROMPT)
SUMQ = ("Now explain the key idea above to a non-expert in AT MOST 12 WORDS. "
        "Plain everyday English, no jargon, no equations, no symbols, no "
        "technical terms. A sentence fragment is fine, grammar doesn't matter. "
        "Output ONLY those <=12 words, nothing else.")


def corner_src(words, lead):
    kw = ", ".join(words) if isinstance(words, list) else str(words)
    return f"Keywords: {kw}.\n\n{lead}"


def quarter_bary(frac):
    k = round(frac * R)
    return (R - k, k, 0)


def load_cached(suf):
    path = os.path.join(sc.RESULTS, f"absfollow_{SET}{TAG}{suf}.json")
    return {tuple(c["bary"]): c for c in json.load(open(path))["cells"]}


def gen(model, gen_tok, turns, max_new=30):
    ids = gen_tok.apply_chat_template(turns, tokenize=True, add_generation_prompt=True,
                                      return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                         max_new_tokens=max_new, do_sample=False)
    return gen_tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def cap_words(s, n=12):
    """icae's concatenation-style prompting doesn't reliably obey a word-count
    instruction (no chat-template/role separation) -- hard-truncate as a backstop."""
    words = s.split()
    return " ".join(words[:n]) + ("..." if len(words) > n else "")


def main():
    spec = sc.load_corners(sc.corners_path(SET))
    keys = spec["keys"]
    Vraw = spec["leads"]
    V = {k: corner_src(spec["words"][k], spec["leads"][k]) for k in keys}

    summaries_path = os.path.join(HERE, "keyword_summaries.json")
    all_summaries = json.load(open(summaries_path)) if os.path.exists(summaries_path) else {}
    out = all_summaries.get(SET, {"doc2lora": {}, "incontext": {}, "icae": {}})

    if ONLY == "icae":
        print("ONLY=icae: skipping doc2lora/in-context regeneration, reusing cached", flush=True)
    else:
      # ---- doc2lora + in-context share the qwen base model ----
      model, gen_tok, ctx_tok = load_model(os.environ["DOC2LORA_CKPT"])
      genes = {}
      for k in keys:
        genes[k] = extract_norm_lora_emb(model, ctx_tok, V[k], max_length=512)
        model.reset()

      d_cells = load_cached("")
      for frac in QUARTERS:
        bary = quarter_bary(frac)
        rec = d_cells[bary]
        w = rec["weights"]

        emb = mix_embeddings([genes[keys[0]], genes[keys[1]]], weights=w, mode="full")
        internalize_from_norm_lora_emb(model, emb)
        turns = [{"role": "user", "content": ABS_PROMPT},
                {"role": "assistant", "content": rec["doc2lora"]["abs"]},
                {"role": "user", "content": KWQ},
                {"role": "assistant", "content": rec["doc2lora"]["kw"]},
                {"role": "user", "content": SUMQ}]
        s = gen(model, gen_tok, turns, max_new=30)
        model.reset()
        out["doc2lora"][str(frac)] = s
        print("doc2lora", frac, "->", s, flush=True)

        inc_first = INC_ABS2.format(A=Vraw[keys[0]], B=Vraw[keys[1]],
                                    pa=round(w[0] * 100), pb=round(w[1] * 100))
        turns = [{"role": "user", "content": inc_first},
                {"role": "assistant", "content": rec["incontext"]["abs"]},
                {"role": "user", "content": KWQ},
                {"role": "assistant", "content": rec["incontext"]["kw"]},
                {"role": "user", "content": SUMQ}]
        s = gen(model, gen_tok, turns, max_new=30)
        out["incontext"][str(frac)] = s
        print("incontext", frac, "->", s, flush=True)

      del model
      torch.cuda.empty_cache()

    # ---- icae: separate mistral-based model, concatenation-style prompting
    # (matches decode_absfollow_icae.py's own pattern -- no chat-template turns) ----
    import icae_slots as ici
    from decode_absfollow_icae import decode_sentence

    imodel = ici.load_model()
    with torch.no_grad():
        slots = {k: ici.compress(imodel, corner_src(spec["words"][k], spec["leads"][k])) for k in keys}
    m = min(s.shape[0] for s in slots.values())
    S = {k: slots[k][:m] for k in keys}

    i_cells = load_cached("_icae")
    for frac in QUARTERS:
        bary = quarter_bary(frac)
        rec = i_cells[bary]
        w = rec["weights"]
        mixed = w[0] * S[keys[0]] + w[1] * S[keys[1]]
        ab, kw = rec["icae"]["abs"], rec["icae"]["kw"]
        with torch.no_grad():
            s = decode_sentence(imodel, mixed, f"{ab}\n\n{kw}\n\n{SUMQ}", max_new=30)
        s = cap_words(s)
        out["icae"][str(frac)] = s
        print("icae", frac, "->", s, flush=True)

    all_summaries[SET] = out
    json.dump(all_summaries, open(summaries_path, "w"), indent=2)
    print("wrote", summaries_path)


if __name__ == "__main__":
    main()
