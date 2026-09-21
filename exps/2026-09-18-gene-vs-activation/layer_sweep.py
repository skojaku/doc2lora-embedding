"""Pick the activation arm's configuration by measurement, not by hand (#152).

`diag_prompts.py` showed only the `continue` family carries any signal, but on two
documents by eye. This quantifies it across layer x slot count x pooling mode so the
activation arm runs at ITS best setting. Fixing the configuration by hand is how a
baseline gets crippled (survey pattern P2), and this experiment exists to avoid that.

Two metrics, both scored against the floor:

  fidelity   SBERT cos(decode, source abstract), reported as a DELTA over the floor
             decode (the same prompt with nothing injected). `continue` has a strong
             prior toward biomedical abstracts, so absolute cosine is meaningless --
             only the lift over what the prompt alone produces counts.
  MRR        rank of the true source among all N abstracts by that cosine. This is the
             discriminative measure and the one that matters: a decode can look
             on-topic and still fail to identify its own source.

Batching: every document shares the same decode prompt and differs only in the injected
vector, so a batch is one `generate` call with a [B, n_slots, d] patch.

  CUDA_VISIBLE_DEVICES=1 python layer_sweep.py [--n 50]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import act_common as A  # noqa: E402
from act_common import load_qwen, extract_acts, _patch_at, load_aps_text  # noqa: E402

# the one family that carried signal; the other three are dead (see NOTE.md)
CONTINUE_TPL = "The following is a scientific abstract.\n\n{MARK}"
SBERT_ID = "sentence-transformers/all-mpnet-base-v2"


def build(Q, n_slots):
    """Continuation prompt with exactly n_slots patchable positions (no chat template)."""
    tok = Q["tok"]
    head, tail = CONTINUE_TPL.split("{MARK}", 1)
    head_ids = tok(head, add_special_tokens=False)["input_ids"]
    tail_ids = tok(tail, add_special_tokens=False)["input_ids"]
    slot_id = tok(A.PLACEHOLDER, add_special_tokens=False)["input_ids"][0]
    ids = torch.tensor([head_ids + [slot_id] * n_slots + tail_ids], dtype=torch.long)
    return ids, list(range(len(head_ids), len(head_ids) + n_slots))


@torch.no_grad()
def decode_batch(Q, vecs, layer, n_slots, max_new_tokens=64, batch_size=16):
    """Decode many activation vectors through one shared prompt."""
    ids, pos = build(Q, n_slots)
    tok, model = Q["tok"], Q["model"]
    outs = []
    for i in range(0, len(vecs), batch_size):
        v = vecs[i:i + batch_size]                                  # [b, d]
        b = v.shape[0]
        vv = v.unsqueeze(1).repeat(1, len(pos), 1)                  # [b, n_slots, d]
        batch_ids = ids.repeat(b, 1).to(Q["device"])
        with _patch_at(model, layer, pos, vv):
            out = model.generate(input_ids=batch_ids, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        outs += [tok.decode(o[batch_ids.shape[1]:], skip_special_tokens=True).strip()
                 for o in out]
    return outs


@torch.no_grad()
def floor_decode(Q, n_slots, max_new_tokens=64):
    ids, _ = build(Q, n_slots)
    out = Q["model"].generate(input_ids=ids.to(Q["device"]), max_new_tokens=max_new_tokens,
                              do_sample=False, pad_token_id=Q["tok"].eos_token_id)
    return Q["tok"].decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def score(sb, decodes, abstracts, floor_text):
    """fidelity delta over floor + MRR of the true source."""
    D = sb.encode(decodes, normalize_embeddings=True, show_progress_bar=False)
    S = sb.encode(abstracts, normalize_embeddings=True, show_progress_bar=False)
    f = sb.encode([floor_text], normalize_embeddings=True, show_progress_bar=False)[0]
    diag = (D * S).sum(1)                       # cos(decode_i, abstract_i)
    floor_cos = S @ f                           # cos(floor, abstract_i)
    C = D @ S.T                                 # [N, N]
    ranks = (C > diag[:, None]).sum(1) + 1      # rank of the true source
    return dict(
        fidelity=float(diag.mean()),
        floor_fidelity=float(floor_cos.mean()),
        fidelity_delta=float(diag.mean() - floor_cos.mean()),
        mrr=float((1.0 / ranks).mean()),
        top1=float((ranks == 1).mean()),
        chance_mrr=float(np.mean([1 / r for r in range(1, len(decodes) + 1)])),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12, 15, 18, 24, 30, 35])
    ap.add_argument("--slots", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--modes", nargs="+", default=["mean", "last"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=str(HERE / "results/layer_sweep.json"))
    a = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    Q = load_qwen()
    docs = load_aps_text(n=a.n, seed=3)
    abstracts = list(docs.abstract)
    print(f"[data] {len(docs)} APS abstracts", flush=True)

    sb = SentenceTransformer(SBERT_ID, device="cpu")     # keep the GPU for Qwen
    floors = {s: floor_decode(Q, s) for s in a.slots}
    for s, t in floors.items():
        print(f"[floor slots={s}] {t[:110]}", flush=True)

    rows = []
    for mode in a.modes:
        for layer in a.layers:
            V = extract_acts(Q, list(docs.text), layer, mode=mode)
            for n_slots in a.slots:
                dec = decode_batch(Q, V, layer, n_slots, batch_size=a.batch_size)
                sc = score(sb, dec, abstracts, floors[n_slots])
                row = dict(mode=mode, layer=layer, slots=n_slots, **sc,
                           sample_decode=dec[0][:200])
                rows.append(row)
                print(f"  mode={mode:<4} L{layer:<2} slots={n_slots}: "
                      f"fid {sc['fidelity']:.3f} (floor {sc['floor_fidelity']:.3f}, "
                      f"delta {sc['fidelity_delta']:+.3f})  "
                      f"MRR {sc['mrr']:.3f} (chance {sc['chance_mrr']:.3f})  "
                      f"top1 {sc['top1']:.2f}", flush=True)

    rows.sort(key=lambda r: -r["mrr"])
    print("\n[best by MRR]")
    for r in rows[:5]:
        print(f"  mode={r['mode']} L{r['layer']} slots={r['slots']}: "
              f"MRR {r['mrr']:.3f}  delta {r['fidelity_delta']:+.3f}")
        print(f"     {r['sample_decode'][:160]!r}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"config": vars(a), "floors": floors, "rows": rows},
                                      indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
