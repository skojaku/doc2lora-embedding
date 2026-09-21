"""LatentQA arm: a TRAINED activation decoder, as the baseline's ceiling (#152).

The obvious objection to the activation result so far: raw activations decode badly
*under training-free patching*, but Patchscopes is not the only way to read them. LatentQA
(Pan, Chen & Steinhardt, arXiv:2412.08686) finetunes a decoder LLM on (activation,
question, answer) triples, so it is a purpose-built activation reader. If activations are
decodable at all, this is where it should show.

It is therefore NOT a like-for-like arm and must not be reported as one. Everything else
in #152 is training-free on both sides; this arm gives the activation side a decoder
trained for exactly this job while \\doctolora gets none. Read it as **the ceiling raw
activations reach when someone trains a reader for them**, and a gene win here is
correspondingly stronger than a gene win against patching.

Setup, as released:
  target  : Llama-3-8B-Instruct (NousResearch mirror -- `meta-llama` is gated on our
            token and the weights are the same re-upload)
  decoder : aypan17/latentqa_llama-3-8b-instruct, a r=16 LoRA on that same base
  read    : layer 15 of the target (their `min_layer_to_read` default -- the same depth
            our own Qwen sweep independently picked as best)
  write   : layer 0 of the decoder

Two limits to state rather than paper over:
  1. Different base model. Every other arm runs on Qwen3-4B; this one runs on Llama-3-8B
     because that is what the released decoder was trained for. Model and method are
     therefore confounded in this arm alone.
  2. LatentQA was trained on persona/behaviour questions about an assistant, not on
     document-content questions. Asking it what a document is about is off-distribution
     for it, in the same way feeding T2L a paper abstract is off-distribution there (#151).
     That is a real limit on what a low score would mean, and it is why the arm is a
     ceiling probe and not a verdict.

Writes decodes in the shape `score_compare.py` reads, so M-A and M-B are computed by the
same code, against the same 99 same-subfield distractors, as every other arm.

  CUDA_VISIBLE_DEVICES=0,1 python latentqa_arm.py --n 500
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

LQA_REPO = os.environ.get("LATENTQA_REPO", str(ROOT / "latentqa"))
TARGET = os.environ.get("LATENTQA_TARGET", "NousResearch/Meta-Llama-3-8B-Instruct")
# the name LatentQA's tables are keyed by; the mirror carries the same weights
CANONICAL_TARGET = "meta-llama/Meta-Llama-3-8B-Instruct"
DECODER = os.environ.get(
    "LATENTQA_DECODER",
    str(ROOT / "data/agent_assets/hf_cache/hub/models--aypan17--latentqa_llama-3-8b-instruct"
        "/snapshots/771bc02bc54c33fdd17fdad1db394958933b8ca4"))
os.environ.setdefault("HF_HOME", str(ROOT / "data/agent_assets/hf_cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# One content question, the same one the other arms are asked.
QUESTIONS = [["What is the specific topic of this document? Describe its content."]]


def build_args(batch_size):
    sys.path.insert(0, LQA_REPO)
    from lit.configs.interpret_config import interpret_config

    a = interpret_config()
    a.target_model_name = TARGET
    a.decoder_model_name = DECODER
    a.batch_size = batch_size
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=16, help="documents per interpret() call")
    ap.add_argument("--out", default=str(HERE / "results/decodes_latentqa.json"))
    a = ap.parse_args()

    sys.path.insert(0, LQA_REPO)
    from transformers import PreTrainedModel

    # Upstream inconsistency at latentqa@a2dcb6f: `lit/utils/dataset_utils.py` renamed
    # `tokenize` to `lqa_tokenize` but `lit/reading.py` still imports the old name, so
    # importing reading.py fails outright. Alias it back before the import rather than
    # editing their tree -- the signature is unchanged, so this is a rename shim and
    # nothing more.
    import lit.utils.dataset_utils as _du

    if not hasattr(_du, "tokenize"):
        _du.tokenize = _du.lqa_tokenize

    # LatentQA keys several module-level tables by the canonical model name
    # (PAD_TOKEN_IDS, the chat-template tables, and others), and we load the weights
    # from the ungated NousResearch mirror, whose name misses every one of them. Alias
    # the canonical key onto the mirror name in EVERY such table rather than the two we
    # happened to trip over first -- a missed table would not raise, it would silently
    # fall back to a different chat template and quietly change what the decoder reads.
    _aliased = []
    for _name, _obj in list(vars(_du).items()):
        if isinstance(_obj, dict) and CANONICAL_TARGET in _obj:
            _obj[TARGET] = _obj[CANONICAL_TARGET]
            _aliased.append(_name)
    print(f"[shim] aliased {CANONICAL_TARGET} -> {TARGET} in {_aliased}", flush=True)

    from lit.reading import interpret, ForCausalLMLossPatched
    from lit.utils.infra_utils import get_model, get_tokenizer

    from decode_compare import sample_docs

    df = sample_docs(a.n)          # identical sample + seed as every other arm
    print(f"[data] {len(df)} papers across {df.subdivision.nunique()} subdivisions",
          flush=True)

    args = build_args(a.batch_size)
    PreTrainedModel.loss_function = staticmethod(ForCausalLMLossPatched)
    tokenizer = get_tokenizer(TARGET)
    n_gpu = torch.cuda.device_count()
    dec_dev = "cuda:1" if n_gpu > 1 else "cuda:0"
    print(f"[models] target=cuda:0  decoder={dec_dev}  (read L{args.min_layer_to_read} "
          f"-> write L{args.layer_to_write})", flush=True)
    decoder_model = get_model(TARGET, tokenizer, load_peft_checkpoint=DECODER, device=dec_dev)
    target_model = get_model(TARGET, tokenizer, device="cuda:0")

    decodes, floor = [], None
    for i in range(0, len(df), a.chunk):
        chunk = list(df.text[i:i + a.chunk])
        qa, _, _ = interpret(target_model, decoder_model, tokenizer,
                             [[t] for t in chunk], QUESTIONS, args, generate=True)
        for t in chunk:
            pairs = qa.get(t, [])
            decodes.append(pairs[0][1].strip() if pairs else "")
        print(f"  [{min(i + a.chunk, len(df))}/{len(df)}] {decodes[-1][:100]!r}", flush=True)

    # floor: the same question with an EMPTY document, i.e. what the decoder says when
    # the activations carry nothing. Every score is a lift over this, never absolute.
    qa, _, _ = interpret(target_model, decoder_model, tokenizer, [[""]], QUESTIONS, args,
                         generate=True)
    fp = qa.get("", [])
    floor = fp[0][1].strip() if fp else ""

    rec = dict(
        arm="latentqa", n=len(df),
        config={"target": TARGET, "decoder": DECODER,
                "read_layer": args.min_layer_to_read, "write_layer": args.layer_to_write,
                "trained_decoder": True, "base_model_differs_from_other_arms": True},
        floor=floor,
        items=[{"paper_id": int(p), "subdivision": s, "title": t, "abstract": ab,
                "decode": d}
               for p, s, t, ab, d in zip(df.paper_id, df.subdivision, df.title,
                                         df.abstract, decodes)],
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=2))
    print(f"[floor] {floor[:160]}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
