"""Verify the generated LoRA is actually reaching the model (issue #151).

`sanity_degeneracy.py` shows T2L adapters decode to unrelated text. Before that can be
reported, we must rule out the boring explanation: that `set_peft_model_state_dict`
silently no-ops on a key mismatch and we are reading the BASE model the whole time.

Checks, in order of how damning a failure would be:
  1. key match     -- what set_peft_model_state_dict reports as missing/unexpected
  2. weight change -- do the model's lora_A/lora_B tensors actually change value
  3. logit change  -- does the next-token distribution move vs the zeroed adapter
  4. text change   -- base vs doc-A vs doc-B decode of the same prompt

Run:
  export PYTHONPATH=$T2L_SRC
  CUDA_VISIBLE_DEVICES=0 python verify_wiring.py
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
from bench_data import out_dir   # bench_data.py sits next to this file
DATA = out_dir("t2l")           # where this chain writes

sys.path.insert(0, str(HERE))
from t2l_common import load_t2l, text_to_lora, apply_lora, decode, load_aps_text  # noqa: E402

TOPIC_PROMPT = "In one sentence, what is the specific topic of this document?"
PROBE = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"


def lora_param_snapshot(model):
    return {n: p.detach().float().clone() for n, p in model.named_parameters() if "lora_" in n}


@torch.no_grad()
def next_token_logits(T, prompt):
    tok, model = T["tokenizer"], T["model"]
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt").to(model.device)
    return model(input_ids=ids).logits[0, -1].float().cpu()


def main():
    T = load_t2l()
    model = T["model"]
    res = {}

    lora_names = [n for n, _ in model.named_parameters() if "lora_" in n]
    print(f"[1] model has {len(lora_names)} lora params; e.g. {lora_names[0]}")

    docs = load_aps_text(n=2, seed=7)
    e0a, e1a, sd_a = text_to_lora(T, docs.text[0])
    e0b, e1b, sd_b = text_to_lora(T, docs.text[1])
    print(f"    generated sd has {len(sd_a)} entries; e.g. {next(iter(sd_a))}")
    res["n_model_lora_params"] = len(lora_names)
    res["n_generated_entries"] = len(sd_a)
    res["key_overlap"] = len(set(sd_a) & set(lora_names))
    print(f"    key overlap with model params: {res['key_overlap']}")

    # ---- zero the adapter -> "base" reference -------------------------------
    zero_sd = {k: torch.zeros_like(v) for k, v in sd_a.items()}
    apply_lora(T, zero_sd)
    before = lora_param_snapshot(model)
    logits_base = next_token_logits(T, TOPIC_PROMPT)
    text_base = decode(T, TOPIC_PROMPT, max_new_tokens=40)

    # ---- apply doc A ---------------------------------------------------------
    info = apply_lora(T, sd_a)
    after = lora_param_snapshot(model)
    missing = list(getattr(info, "missing_keys", []) or [])
    unexpected = list(getattr(info, "unexpected_keys", []) or [])
    res["missing_keys"] = missing[:5]
    res["n_missing"] = len(missing)
    res["unexpected_keys"] = unexpected[:5]
    res["n_unexpected"] = len(unexpected)
    print(f"[2] set_peft_model_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")

    changed = sum(1 for n in before if not torch.equal(before[n], after[n]))
    res["n_lora_params_changed"] = changed
    print(f"    lora params changed after apply: {changed}/{len(before)}")
    if PROBE in before:
        d = (after[PROBE] - before[PROBE]).abs().max().item()
        res["probe_max_abs_delta"] = d
        print(f"    {PROBE}\n      max|delta| = {d:.6g}")

    logits_a = next_token_logits(T, TOPIC_PROMPT)
    text_a = decode(T, TOPIC_PROMPT, max_new_tokens=40)

    # ---- apply doc B ---------------------------------------------------------
    apply_lora(T, sd_b)
    logits_b = next_token_logits(T, TOPIC_PROMPT)
    text_b = decode(T, TOPIC_PROMPT, max_new_tokens=40)

    def dlog(x, y):
        return dict(max_abs=float((x - y).abs().max()),
                    mean_abs=float((x - y).abs().mean()),
                    argmax_same=bool(x.argmax() == y.argmax()))

    res["logits"] = {"base_vs_A": dlog(logits_base, logits_a),
                     "base_vs_B": dlog(logits_base, logits_b),
                     "A_vs_B": dlog(logits_a, logits_b)}
    print("[3] next-token logits")
    for k, v in res["logits"].items():
        print(f"    {k:10s} max|d| {v['max_abs']:.4f}  mean|d| {v['mean_abs']:.4f}  "
              f"same argmax: {v['argmax_same']}")

    res["texts"] = {"base(zeroed adapter)": text_base,
                    f"docA: {docs.title[0][:60]}": text_a,
                    f"docB: {docs.title[1][:60]}": text_b}
    print("[4] decodes")
    for k, v in res["texts"].items():
        print(f"    {k}\n      -> {v}")

    out = DATA / "verify_wiring.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
