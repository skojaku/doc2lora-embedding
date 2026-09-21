"""Generative Adapter (Chen et al., ICLR 2025) — Arm 2 of the baseline work (issue #151).

Arm 1 (Text-to-LoRA) showed a hypernetwork that generates adapters and has no *learned*
space. Generative Adapter is the other half of the argument: it generates adapters from
the document's full per-token hidden states, so it plausibly carries content — but there
is **no fixed-size bottleneck** anywhere in it to store, index or compare. Together the
two arms say that generating adapters is not sufficient; a compressed, fixed-size, learned
space is what the claim rests on.

`sections/01-intro.tex` currently excludes it on storage, quoting the paper: two
4,096x128 submatrices per layer, about 32M parameters per document for Mistral-7B. **That
number is cited, not measured by us.** This arm converts the citation into a measurement,
which is the same move Arm 1 made for the prose argument about T2L.

Base model is `Generative-Adapter-Mistral-7B-Instruct-v0.2` — the same
Mistral-7B-Instruct-v0.2 as the ICAE and T2L arms, so decodes are comparable at the
generator level.

Three things, all small-scale by design (no retrieval arm: the whole point is that it
cannot be indexed):

  storage   generate one adapter, write it, measure the actual bytes
  nodes     the 28 PACS node means -- feasible only if the generated adapter has a
            FIXED shape across documents, which `--probe` checks first and which is the
            crux of the storage argument either way
  midpoints the same 100 corner pairs as section 4.3 and #152

Reuses the existing loader in `exps/2026-05-26-benchmark/scripts/decode/decode_ga.py`,
including its peft-version shims, rather than re-deriving it.

  CUDA_VISIBLE_DEVICES=1 python ga_arm.py --probe
  CUDA_VISIBLE_DEVICES=1 python ga_arm.py --stage nodes --cap 64
  CUDA_VISIBLE_DEVICES=1 python ga_arm.py --stage midpoints --n 50
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
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
os.environ.setdefault("HF_HOME", str(ROOT / "data/agent_assets/hf_cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

GA_REPO = str(ROOT / "exps/2026-05-26-baseline/generative-adapter")
GA_MODEL = "generative-adaptor/Generative-Adapter-Mistral-7B-Instruct-v0.2"
KG = ROOT / "exps/2026-07-02-simplex-kwgrid"
CA = ROOT / "exps/2026-05-28-concept-analogy-aps"
WINDOW, MERGE = 1024, "sequential"

sys.path.insert(0, str(ROOT / "exps/2026-06-11-baseline-trees"))
from shared_prompt import LABEL_PROMPT  # noqa: E402

CONT_PROMPT = "The following is a scientific abstract. Write it."
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


def load_ga(device="cuda"):
    """Load Generative Adapter via the existing loader's shims (peft version drift)."""
    sys.path.insert(0, str(ROOT / "exps/2026-05-26-benchmark/scripts/decode"))
    sys.path.insert(0, GA_REPO)
    import json as _json

    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from fastlora.config import FastLoraConfig
    from fastlora.model import (FastLoraModelForCausalLM, FastLoraModel, FastLoraLinear,
                                get_peft_model_state_dict, set_peft_model_state_dict)

    _orig = FastLoraLinear.set_adapter
    FastLoraLinear.set_adapter = lambda self, names, inference_mode=None, **_k: _orig(self, names)
    import peft.mapping as peft_mapping
    import peft.peft_model as peft_model
    tmap = getattr(peft_model, "PEFT_TYPE_TO_TUNER_MAPPING",
                   getattr(peft_model, "PEFT_TYPE_TO_MODEL_MAPPING", None))
    if tmap is not None:
        tmap["FASTLORA"] = FastLoraModel
    peft_mapping.PEFT_TYPE_TO_CONFIG_MAPPING.update({"FASTLORA": FastLoraConfig})
    peft_model.get_peft_model_state_dict = get_peft_model_state_dict
    peft_model.set_peft_model_state_dict = set_peft_model_state_dict

    cfg = _json.load(open(hf_hub_download(GA_MODEL, "adapter_config.json")))
    base_path = cfg.get("base_model_name_or_path")
    cfg["task_type"] = "CAUSAL_LM"
    pc = FastLoraConfig(**{k: v for k, v in cfg.items()
                           if k in FastLoraConfig.__dataclass_fields__})
    pc.task_type = "FAST_LORA_CAUSAL_LM"
    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa")
    model = FastLoraModelForCausalLM.from_pretrained(base, GA_MODEL, adapter_name="default",
                                                     is_trainable=False, config=pc).to(device)
    tok = AutoTokenizer.from_pretrained(GA_MODEL)
    from fastlora.eval_utils import fastlora_generate_adaptor, fastlora_conditional_generate
    return dict(model=model, tok=tok, base=base_path,
                gen=fastlora_generate_adaptor, dec=fastlora_conditional_generate)


def adapt(G, text):
    return G["gen"](G["model"], G["tok"], text, merge_strategy=MERGE, max_window_size=WINDOW)


def decode(G, w, prompt, max_new_tokens=64):
    return G["dec"](G["model"], G["tok"], input_text=prompt, use_chat=True,
                    mode="weights", lora_weights=w, max_new_tokens=max_new_tokens).strip()


def flat(w, prefix=""):
    """fastlora_generate_adaptor returns a NESTED dict {module: {param: tensor}}; flatten
    it to {"module.param": tensor} for shape, size and save bookkeeping."""
    out = {}
    for k, v in w.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


def mean_weights(ws, weights=None):
    """Elementwise mean of adapter weight dicts, preserving the nesting.

    Valid only if the shapes are fixed across documents -- `run_probe` checks that first,
    because if they are not, a mean is not even defined for this method.
    """
    if weights is None:
        weights = [1.0 / len(ws)] * len(ws)

    def rec(nodes):
        first = nodes[0]
        if isinstance(first, dict):
            return {k: rec([n[k] for n in nodes]) for k in first}
        acc = torch.zeros_like(first, dtype=torch.float32)
        for wt, v in zip(weights, nodes):
            acc += float(wt) * v.float()
        return acc.to(first.dtype)

    return rec(ws)


class WeightAccumulator:
    """Streaming fp32 mean of adapter weight dicts, preserving nesting.

    Necessary, not tidy: one Generative Adapter is 268M parameters = 1.07 GB fp32
    (measured, see run_probe), so holding a node's members in a list before averaging
    would need 64 GB for cap=64. Accumulate on CPU and keep exactly one running sum.
    """

    def __init__(self):
        self.acc, self.n = None, 0.0

    def add(self, w, weight=1.0):
        def rec(dst, src):
            if isinstance(src, dict):
                if dst is None:
                    dst = {}
                for k, v in src.items():
                    dst[k] = rec(dst.get(k), v)
                return dst
            v = src.detach().float().cpu() * float(weight)
            return v if dst is None else dst + v

        self.acc = rec(self.acc, w)
        self.n += weight
        return self

    def mean(self, like):
        assert self.n > 0

        def rec(a, ref):
            if isinstance(ref, dict):
                return {k: rec(a[k], ref[k]) for k in ref}
            return (a / self.n).to(device=ref.device, dtype=ref.dtype)

        return rec(self.acc, like)


def n_params(w):
    return sum(int(np.prod(tuple(v.shape))) for v in flat(w).values())


# --------------------------------------------------------------------------- #
def run_probe(G, out_path):
    texts = {
        "short": "Superconductivity in cuprates.",
        "medium": ("Superconductivity in cuprates. We study electron pairing and the "
                   "critical temperature of copper-oxide superconductors across doping."),
        "long": ("Superconductivity in cuprates. " + "We study electron pairing and the "
                 "critical temperature of copper-oxide superconductors across doping. " * 24),
    }
    info, shapes = {}, {}
    for name, t in texts.items():
        w = adapt(G, t)
        fw = flat(w)
        sh = {k: tuple(v.shape) for k, v in fw.items()}
        shapes[name] = sh
        tmp = Path(out_path).parent / f"_ga_probe_{name}.npz"
        np.savez(tmp, **{k: v.detach().cpu().float().numpy() for k, v in fw.items()})
        info[name] = {"n_tokens": len(G["tok"](t)["input_ids"]), "n_tensors": len(fw),
                      "n_params": n_params(w), "npz_bytes": tmp.stat().st_size}
        tmp.unlink()
        print(f"  [{name:6s}] {info[name]['n_tokens']:>5} tok -> {info[name]['n_tensors']} tensors, "
              f"{info[name]['n_params']:,} params, {info[name]['npz_bytes']/1e6:.1f} MB fp32",
              flush=True)
    fixed = len({json.dumps(s, sort_keys=True) for s in shapes.values()}) == 1
    print(f"\n[fixed shape across document lengths] {fixed}")
    if not fixed:
        print("  -> adapter size depends on the document; node means are NOT defined and "
              "the storage argument is even stronger than the cited figure")
    ref = list(shapes.values())[0]
    per_doc = info["medium"]
    print(f"[storage] {per_doc['n_params']:,} params/document "
          f"= {per_doc['n_params']*2/1e6:.1f} MB fp16 / {per_doc['n_params']*4/1e6:.1f} MB fp32")
    Path(out_path).write_text(json.dumps(
        {"fixed_shape": fixed, "per_text": info,
         "example_shapes": {k: list(v) for k, v in ref.items()}}, indent=2))
    print(f"wrote {out_path}")
    return fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["probe", "nodes", "midpoints"], default="probe")
    ap.add_argument("--cap", type=int, default=64, help="members averaged per node")
    ap.add_argument("--n", type=int, default=50, help="pairs per stratum")
    a = ap.parse_args()
    res = HERE / "results"
    res.mkdir(parents=True, exist_ok=True)

    G = load_ga()
    print(f"[loaded] {GA_MODEL} on {G['base']}", flush=True)

    if a.stage == "probe":
        run_probe(G, res / "ga_probe.json")
        return

    if a.stage == "nodes":
        pg = pd.read_parquet(CA / "results/paper_groups.parquet")
        txt = pd.read_parquet(ROOT / "data/aps/paper_text_pid.parquet",
                              columns=["paper_id", "title", "abstract"])
        txt = txt[txt.abstract.notna()]
        txt = txt.assign(title=txt.title.fillna("").str.strip())
        id2t = dict(zip(txt.paper_id.astype(int),
                        (txt.title + ". " + txt.abstract.str.strip()).str.strip(". ")))
        rng = np.random.default_rng(0)
        out = {}
        for code, level, val in node_list():
            col = {"main": "main_class_id", "division": "division"}.get(level, "subdivision")
            m = pg[pg[col] == val].paper_id.values if level == "main" else \
                pg[pg[col].astype(str) == str(val)].paper_id.values
            ids = np.array([int(p) for p in m if int(p) in id2t])
            if len(ids) > a.cap:
                ids = rng.choice(ids, a.cap, replace=False)
            acc, last = WeightAccumulator(), None
            for pid in ids:                       # stream: never hold two adapters at once
                last = adapt(G, id2t[int(pid)])
                acc.add(last)
            mw = acc.mean(last)
            out[code] = {"level": level, "n_used": int(len(ids)),
                         "ga_label": decode(G, mw, LABEL_PROMPT, 24),
                         "ga_cont": decode(G, mw, CONT_PROMPT, 64)}
            (res / "ga_nodes.json").write_text(json.dumps(out, indent=2))
            print(f"  [{code:<6} n={len(ids):>4}] {out[code]['ga_label']!r}", flush=True)
        return

    # midpoints
    pairs = []
    for L in ("L1", "L5"):
        for i in range(a.n):
            p = KG / f"corners_pair{L}_{i:02d}.json"
            if p.exists():
                c = json.loads(p.read_text())["corners"]
                k = list(c)
                pairs.append({"set": f"{L}_{i:02d}", "stratum": L,
                              "A": c[k[0]]["lead"], "B": c[k[1]]["lead"],
                              "A_name": c[k[0]]["name"], "B_name": c[k[1]]["name"]})
    out = []
    for j, p in enumerate(pairs):
        wa, wb = adapt(G, p["A"]), adapt(G, p["B"])
        d = {str(al): decode(G, mean_weights([wa, wb], [1 - al, al]), CONT_PROMPT, 64)
             for al in (0.0, 0.5, 1.0)}
        out.append({**p, "decodes": d})
        (res / "midpoints_ga.json").write_text(
            json.dumps({"arm": "ga", "alphas": [0.0, 0.5, 1.0], "pairs": out}, indent=2))
        if j % 10 == 0:
            print(f"  [{j}/{len(pairs)}] mid={d['0.5'][:90]!r}", flush=True)


if __name__ == "__main__":
    main()
