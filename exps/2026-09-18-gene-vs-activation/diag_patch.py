"""Why does patching have no visible effect? (#152)

The first smoke test produced decodes identical to the no-patch floor ("the document is
empty"). Before concluding anything about activations, rule out the mechanical causes,
in order:

  1. does the pre-hook fire at all, and does the tensor it returns actually reach the block
  2. is the placeholder position the token we think it is
  3. is the injected vector in-distribution in MAGNITUDE -- mean-pooling shrinks a vector
     well below the norm of a real hidden state at that layer, and a too-small vector is
     simply ignored
  4. is one patched position enough leverage over a 24-token prompt

(3) and (4) are baseline-fairness issues, not cosmetics: getting them wrong would hand
us a "raw activations do not decode" result that is really "we injected a vector the
model could not read".

  CUDA_VISIBLE_DEVICES=1 python diag_patch.py
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from act_common import (  # noqa: E402
    load_qwen, extract_acts, build_decode_inputs, _blocks, _patch_at,
    load_aps_text, TOPIC_PROMPT, patch_decode, nopatch_decode,
)

LAYERS = [8, 18]


def main():
    Q = load_qwen()
    model, tok = Q["model"], Q["tok"]
    docs = load_aps_text(n=3, seed=1)

    # ---- 2. placeholder position -------------------------------------------
    ids, pos = build_decode_inputs(Q, TOPIC_PROMPT, n_slots=1)
    toks = tok.convert_ids_to_tokens(ids[0])
    print(f"[pos] placeholder at {pos}; token there = {toks[pos[0]]!r}")
    print(f"[pos] context: {toks[max(0,pos[0]-3):pos[0]+3]}")

    # ---- 1. does the hook fire and change the tensor ------------------------
    fired = {}

    def probe(_m, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        fired.setdefault("seen", []).append(tuple(h.shape))
        fired["last_row"] = h[0, pos[0]].detach().float().clone()
        return None

    L = LAYERS[0]
    hdl = _blocks(model)[L].register_forward_pre_hook(probe, with_kwargs=True)
    with torch.no_grad():
        model(input_ids=ids.to(Q["device"]))
    hdl.remove()
    baseline_row = fired["last_row"]
    print(f"[hook] fired on shapes {fired['seen']}")

    # hooks run in registration order, so the probe must be registered INSIDE the
    # patch context to observe the patched tensor rather than the tensor before it
    sentinel = torch.full((1, Q["d"]), 7.0)
    fired.clear()
    with _patch_at(model, L, pos, sentinel):
        hdl = _blocks(model)[L].register_forward_pre_hook(probe, with_kwargs=True)
        with torch.no_grad():
            model(input_ids=ids.to(Q["device"]))
        hdl.remove()
    got = fired["last_row"]
    print(f"[hook] patched row mean={got.mean():.3f} (expect 7.000 if the patch lands); "
          f"unpatched row mean={baseline_row.mean():.3f}")

    # ---- 3. magnitude: real hidden states vs pooled vectors -----------------
    print("\n[magnitude] ||h|| of real prompt positions vs injected vectors")
    with torch.no_grad():
        hs = model(input_ids=ids.to(Q["device"]), output_hidden_states=True).hidden_states
    for L in LAYERS:
        real = hs[L][0].float().norm(dim=-1)
        a_mean = extract_acts(Q, list(docs.text), L, mode="mean")
        a_last = extract_acts(Q, list(docs.text), L, mode="last")
        print(f"  layer {L:>2}: prompt tokens ||h|| median {real.median():.1f} "
              f"[{real.min():.1f}, {real.max():.1f}] | placeholder {real[pos[0]]:.1f} | "
              f"act_mean {a_mean.norm(dim=1).mean():.1f} | act_last {a_last.norm(dim=1).mean():.1f}")

    # ---- 4. leverage: slots x norm-matching --------------------------------
    print("\n[leverage] decodes under slot count and norm matching")
    print(f"  FLOOR (no patch): {nopatch_decode(Q)[:110]}")
    L = 18
    A = extract_acts(Q, list(docs.text), L, mode="mean")
    target = hs[L][0].float().norm(dim=-1).median().item()
    for n_slots in (1, 8):
        for scale_to_target in (False, True):
            v = A[0].clone()
            if scale_to_target:
                v = v / v.norm() * target
            ids_n, pos_n = build_decode_inputs(Q, TOPIC_PROMPT, n_slots=n_slots)
            vv = v.unsqueeze(0).repeat(len(pos_n), 1)
            with _patch_at(model, L, pos_n, vv), torch.no_grad():
                out = model.generate(input_ids=ids_n.to(Q["device"]), max_new_tokens=48,
                                     do_sample=False, pad_token_id=tok.eos_token_id)
            txt = tok.decode(out[0, ids_n.shape[1]:], skip_special_tokens=True).strip()
            print(f"  slots={n_slots} norm_match={int(scale_to_target)}: {txt[:110]}")
    print(f"\n  (source doc: {docs.title[0]})")


if __name__ == "__main__":
    main()
