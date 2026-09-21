"""Which target prompt actually verbalizes a patched activation? (#152)

`diag_patch.py` established that the patch lands (sentinel reproduces exactly), that the
placeholder index is right, and that the injected vector is not too small (act_mean norm
46 vs a prompt-token median of 33). Yet the decode still answers "the document is
empty".

That points at the TARGET PROMPT, not the mechanism. In Patchscopes the target prompt is
the method: a question like "what is the topic of this document?" hands the model an
explicit escape hatch ("it is empty"), whereas the published target prompts are
identity/repetition patterns with no such escape, which force the model to verbalize
whatever sits at the patched position.

Choosing the prompt badly would manufacture a "raw activations do not decode" result
that is really "we asked in a form the method never uses" -- the crippled-baseline
failure (survey pattern P2) that we are running this experiment to avoid. So the prompt
family is swept, and the activation arm gets whichever wins.

Swept: 4 prompt families x layers x slot counts, on 2 documents.

  CUDA_VISIBLE_DEVICES=1 python diag_prompts.py
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import act_common as A  # noqa: E402
from act_common import load_qwen, extract_acts, _blocks, _patch_at, load_aps_text  # noqa: E402

LAYERS = [8, 18, 28]
SLOTS = [1, 8]

# Each family returns (full_prompt_text_with_MARK, uses_chat_template).
# MARK is replaced by the patched slot tokens at exact token positions.
MARK = ""

FAMILIES = {
    # what we tried first: a direct question. Keeps the "it is empty" escape hatch.
    "question": (f"Document: {MARK}\n\nIn one sentence, what is the specific topic of "
                 f"this document?", True),
    # Patchscopes' identity pattern: no escape hatch, the model must emit *something*
    # that corresponds to the patched position.
    "identity": (f"cat -> cat\n1135 -> 1135\nhello -> hello\n{MARK} ->", False),
    # repetition instruction, chat-formatted
    "repeat": (f"Repeat the following text verbatim: {MARK}", True),
    # continuation: no instruction at all, just let the model continue from the slot
    "continue": (f"The following is a scientific abstract.\n\n{MARK}", False),
}


def build(Q, template, use_chat, n_slots):
    """Concatenate token ids so the slot positions are exact (see act_common)."""
    tok = Q["tok"]
    text = template
    if use_chat:
        text = tok.apply_chat_template([{"role": "user", "content": template}],
                                       tokenize=False, add_generation_prompt=True)
    head, tail = text.split(MARK, 1)
    head_ids = tok(head, add_special_tokens=False)["input_ids"]
    tail_ids = tok(tail, add_special_tokens=False)["input_ids"]
    slot_id = tok(A.PLACEHOLDER, add_special_tokens=False)["input_ids"][0]
    ids = torch.tensor([head_ids + [slot_id] * n_slots + tail_ids], dtype=torch.long)
    return ids, list(range(len(head_ids), len(head_ids) + n_slots))


@torch.no_grad()
def run(Q, vec, layer, template, use_chat, n_slots, max_new_tokens=48):
    ids, pos = build(Q, template, use_chat, n_slots)
    vv = torch.as_tensor(vec).reshape(1, -1).repeat(len(pos), 1)
    with _patch_at(Q["model"], layer, pos, vv):
        out = Q["model"].generate(input_ids=ids.to(Q["device"]), max_new_tokens=max_new_tokens,
                                  do_sample=False, pad_token_id=Q["tok"].eos_token_id)
    return Q["tok"].decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def floor(Q, template, use_chat, n_slots, max_new_tokens=48):
    ids, _ = build(Q, template, use_chat, n_slots)
    out = Q["model"].generate(input_ids=ids.to(Q["device"]), max_new_tokens=max_new_tokens,
                              do_sample=False, pad_token_id=Q["tok"].eos_token_id)
    return Q["tok"].decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    Q = load_qwen()
    docs = load_aps_text(n=2, seed=1)
    print("documents:")
    for i in range(len(docs)):
        print(f"  [{i}] {docs.title[i]}")

    acts = {L: extract_acts(Q, list(docs.text), L, mode="mean") for L in LAYERS}

    for fam, (tpl, chat) in FAMILIES.items():
        print(f"\n{'='*78}\n[{fam}]  chat={chat}")
        for n_slots in SLOTS:
            print(f"  -- slots={n_slots} | FLOOR: {floor(Q, tpl, chat, n_slots)[:100]!r}")
            for L in LAYERS:
                for i in range(len(docs)):
                    txt = run(Q, acts[L][i], L, tpl, chat, n_slots)
                    print(f"     L{L:<2} doc{i}: {txt[:120]!r}")


if __name__ == "__main__":
    main()
