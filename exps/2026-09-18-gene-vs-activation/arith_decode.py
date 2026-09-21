"""M-C: decode PACS node MEANS in both representations (#152).

M-A tested single points. This tests the *space* — averaging many documents and
decoding the mean is where the paper's claim actually lives.

Four label sets per node, because a fair comparison needs two different pairings:

  gene_label   mean gene -> the shared LABEL_PROMPT           (the paper's protocol)
  act_label    mean activation -> the same LABEL_PROMPT       (same question, both arms)
  gene_cont    mean gene -> the continuation prompt
  act_cont     mean activation -> the continuation prompt

`*_label` is the direct-question pairing. It is expected to fail for the activation
arm — `diag_prompts.py` showed a direct question lets the model answer "the document is
empty" — which is precisely why `*_cont` exists: the continuation prompt is the only
family activations respond to, so both arms are also run through it and named
afterwards by one off-panel LLM (`arith_name.py`). Reporting only `*_label` would
cripple the baseline; reporting only `*_cont` would drop the protocol the paper uses.

Node membership is `paper_groups.parquet` with CAP=2000 and seed 0 — identical to
`compute_qwen_fullrank_means.py` and to `exps/2026-09-18-t2l-baseline/t2l_label.py`, so
the nodes are paired across all three experiments.

The gene means are NOT recomputed: `exps/2026-06-11-baseline-trees/qwen_fullrank_means.npz`
already holds all 30 nodes at the same CAP and seed.

  CUDA_VISIBLE_DEVICES=0 python arith_decode.py --arm act
  PYTHONPATH=... DOC2LORA_CKPT=... CUDA_VISIBLE_DEVICES=1 python arith_decode.py --arm gene
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "exps/2026-06-11-baseline-trees"))
from shared_prompt import LABEL_PROMPT  # noqa: E402

CA = ROOT / "exps/2026-05-28-concept-analogy-aps"
MEANS_NPZ = ROOT / "exps/2026-06-11-baseline-trees/qwen_fullrank_means.npz"
BEST = dict(layer=15, slots=8, mode="mean")
# The two arms need the SAME instruction delivered differently, because the
# mechanisms differ: a gene is decoded through a chat turn, so the continuation has to
# be phrased as an instruction ("...Write it.") or the model simply echoes the prompt
# back; a patched activation is decoded from a raw completion context, where the same
# sentence is the prefix being continued. Identical wording, different delivery -- the
# asymmetry is in the mechanism, not in what either arm is asked for.
ACT_CONT_PROMPT = "The following is a scientific abstract."
GENE_CONT_PROMPT = "The following is a scientific abstract. Write it."
CAP, SEED = 2000, 0
MAX_NEW_LABEL, MAX_NEW_CONT = 24, 64

SPEC = [
    {"fid": 3, "divs": [("75", ["75.10", "75.30"]), ("71", ["71.10", "71.20"])]},
    {"fid": 0, "divs": [("03", ["03.67", "03.65"]), ("05", ["05.45", "05.40"])]},
    {"fid": 6, "divs": [("11", ["11.15", "11.10"]), ("12", ["12.38", "12.60"])]},
    {"fid": 2, "divs": [("42", ["42.50", "42.65"]), ("47", ["47.27", "47.20"])]},
]


def node_list():
    out = []
    for f in SPEC:
        out.append((str(f["fid"]), "main", f["fid"]))
        for code, subs in f["divs"]:
            out.append((code, "division", code))
            out += [(s, "subdivision", s) for s in subs]
    return out


def load_corpus():
    pg = pd.read_parquet(CA / "results/paper_groups.parquet")
    txt = pd.read_parquet(ROOT / "data/aps/paper_text_pid.parquet",
                          columns=["paper_id", "title", "abstract"])
    txt = txt[txt.abstract.notna()]
    txt = txt.assign(title=txt.title.fillna("").str.strip())
    id2text = dict(zip(txt.paper_id.astype(int),
                       (txt.title + ". " + txt.abstract.str.strip()).str.strip(". ")))
    return pg, id2text


def members(pg, id2text, level, val, rng):
    if level == "main":
        m = pg[pg.main_class_id == val].paper_id.values
    elif level == "division":
        m = pg[pg.division.astype(str) == val].paper_id.values
    else:
        m = pg[pg.subdivision.astype(str) == val].paper_id.values
    ids = np.array([int(p) for p in m if int(p) in id2text])
    if len(ids) > CAP:
        ids = rng.choice(ids, CAP, replace=False)
    return ids


# --------------------------------------------------------------------------- #
def run_act(nodes, pg, id2text):
    from act_common import load_qwen, extract_acts, _patch_at
    from layer_sweep import build

    Q = load_qwen()
    rng = np.random.default_rng(SEED)
    model, tok = Q["model"], Q["tok"]

    def patched(vec, ids, pos, max_new):
        vv = torch.as_tensor(vec).reshape(1, -1).repeat(len(pos), 1)
        with _patch_at(model, BEST["layer"], pos, vv), torch.no_grad():
            out = model.generate(input_ids=ids.to(Q["device"]), max_new_tokens=max_new,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    # label prompt: chat-formatted with a placeholder, exact indices via token ids
    def build_label(n_slots):
        user = f"Document: ⁇\n\n{LABEL_PROMPT}"
        full = tok.apply_chat_template([{"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True)
        head, tail = full.split("⁇", 1)
        h = tok(head, add_special_tokens=False)["input_ids"]
        t = tok(tail, add_special_tokens=False)["input_ids"]
        sid = tok("?", add_special_tokens=False)["input_ids"][0]
        return (torch.tensor([h + [sid] * n_slots + t]),
                list(range(len(h), len(h) + n_slots)))

    lab_ids, lab_pos = build_label(BEST["slots"])
    cont_ids, cont_pos = build(Q, BEST["slots"])   # ACT_CONT_PROMPT, raw completion

    res = {}
    for code, level, val in nodes:
        ids = members(pg, id2text, level, val, rng)
        texts = [id2text[int(p)] for p in ids]
        V = extract_acts(Q, texts, BEST["layer"], mode=BEST["mode"], batch_size=8)
        mean = V.mean(0)
        res[code] = {
            "level": level, "n_used": len(texts),
            "act_label": patched(mean, lab_ids, lab_pos, MAX_NEW_LABEL),
            "act_cont": patched(mean, cont_ids, cont_pos, MAX_NEW_CONT),
        }
        print(f"  [{code:<6} n={len(texts):>5}] label={res[code]['act_label'][:60]!r}\n"
              f"      cont={res[code]['act_cont'][:90]!r}", flush=True)
    return res


def run_gene(nodes):
    """Reuse the stored full-rank means; only the decode is new."""
    sys.path.insert(0, str(ROOT / "libs/doc2lora"))
    import os
    from doc2lora import load_model, decode_adapter

    z = np.load(MEANS_NPZ, allow_pickle=True)
    means = {str(k): z["means"][i] for i, k in enumerate(z["nodes"])}
    n_used = {str(k): int(z["n_used"][i]) for i, k in enumerate(z["nodes"])}
    ckpt = os.environ.get(
        "DOC2LORA_CKPT",
        str(ROOT / "data/agent_assets/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"))
    model, gen_tok, ctx_tok = load_model(ckpt, mode="full")

    res = {}
    for code, level, _ in nodes:
        if code not in means:
            print(f"  [{code}] MISSING from {MEANS_NPZ.name}", flush=True)
            continue
        # the npz stores [L, r, latent]; internalize wants [L, n_modules=1, r, latent]
        # (same reshape as decode_fullrank_field23.py:56)
        emb = torch.from_numpy(means[code]).reshape(36, 1, 8, 512).contiguous()
        res[code] = {
            "level": level, "n_used": n_used[code],
            "gene_label": decode_adapter(model, gen_tok, emb, prompt=LABEL_PROMPT,
                                         max_new_tokens=MAX_NEW_LABEL),
            "gene_cont": decode_adapter(model, gen_tok, emb, prompt=GENE_CONT_PROMPT,
                                        max_new_tokens=MAX_NEW_CONT),
        }
        print(f"  [{code:<6} n={n_used[code]:>5}] label={res[code]['gene_label'][:60]!r}\n"
              f"      cont={res[code]['gene_cont'][:90]!r}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["act", "gene"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or HERE / f"results/arith_nodes_{a.arm}.json")

    nodes = node_list()
    print(f"[nodes] {len(nodes)}", flush=True)
    if a.arm == "act":
        pg, id2text = load_corpus()
        res = run_act(nodes, pg, id2text)
    else:
        res = run_gene(nodes)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
