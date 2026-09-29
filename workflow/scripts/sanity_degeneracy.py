"""Day-1 sanity checks for the Text-to-LoRA arm (issue #151). GPU, ~10 min.

Two questions, both of which must be answered BEFORE any label/fusion result is
interpretable. They are pipeline-validity checks, not comparison arms -- nothing here
goes into a results table.

(A) NON-DEGENERACY. Do different documents produce different adapters at all?
    Pairwise cosine spread in E0 (gte, 1024) / E1 (TaskEncoder, 64) / E2 (generated dW),
    computed separately for our APS abstracts and for T2L's own task descriptions
    (tasks/*/metadata.yaml, i.e. in-distribution inputs). If the abstract adapters are
    near-identical the way ICAE's slots are (cos 0.96-0.99 between unrelated documents),
    that is itself the headline result and the downstream arms shrink accordingly.
    dW cosines use the exact <B1 A1, B2 A2>_F = tr((B1^T B2)(A2 A1^T)) identity, so the
    671M-parameter dense dW is never materialized.

(B) ENDPOINT DECODE. Does a single document's adapter decode back to that document's
    topic? If endpoints do not decode, a midpoint result is a statement about missing
    information, not about geometry, and the manuscript claim has to narrow to match.

Run:
  export PYTHONPATH=$T2L_SRC
  CUDA_VISIBLE_DEVICES=0 python sanity_degeneracy.py [--n-docs 60] [--n-desc 12]
"""
import argparse
import os
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
from bench_data import out_dir   # bench_data.py sits next to this file
DATA = out_dir("t2l")           # where this chain writes

sys.path.insert(0, str(HERE))
from t2l_common import (  # noqa: E402
    ROOT, T2L_SRC, load_t2l, embed_e0, e0_to_e1, e1_to_lora, dw_cos, dw_norm,
    decode_lora, load_aps_text,
)

T2L_REPO = Path(os.environ.get("T2L_REPO", str(Path(T2L_SRC).parent)))
# same verbatim 2-5 word field prompt every other arm gets
sys.path.insert(0, str(ROOT / "data/labels"))
from shared_prompt import LABEL_PROMPT  # noqa: E402

TOPIC_PROMPT = "In one sentence, what is the specific topic of this document?"


def load_task_descriptions(n):
    """T2L's own task descriptions -- in-distribution inputs, for contrast."""
    out = []
    for md in sorted((T2L_REPO / "tasks").glob("*/metadata.yaml")):
        d = yaml.safe_load(md.read_text())
        descs = d.get("descriptions") or []
        if descs:
            out.append((md.parent.name, descs[0]))
        if len(out) >= n:
            break
    return out


def cos_matrix_stats(V):
    """Off-diagonal cosine stats of an [N, D] matrix."""
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    C = Vn @ Vn.T
    iu = np.triu_indices(len(V), k=1)
    v = C[iu]
    return dict(mean=float(v.mean()), sd=float(v.std()),
                p05=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)),
                min=float(v.min()), max=float(v.max()), n_pairs=int(len(v)))


def pairwise_dw_stats(sds, max_pairs=400, seed=0):
    rng = np.random.default_rng(seed)
    pairs = list(combinations(range(len(sds)), 2))
    if len(pairs) > max_pairs:
        pairs = [pairs[i] for i in rng.choice(len(pairs), max_pairs, replace=False)]
    v = np.array([dw_cos(sds[i], sds[j]) for i, j in pairs])
    return dict(mean=float(v.mean()), sd=float(v.std()),
                p05=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)),
                min=float(v.min()), max=float(v.max()), n_pairs=int(len(v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=60)
    ap.add_argument("--n-desc", type=int, default=12)
    ap.add_argument("--n-decode", type=int, default=8)
    ap.add_argument("--out", default=str(DATA / "sanity_degeneracy.json"))
    a = ap.parse_args()

    T = load_t2l()
    print(f"[load] base={T['args'].model_dir}  emb={T['args'].emb_model}  "
          f"layers={len(T['layer_indices'])}", flush=True)

    docs = load_aps_text(n=a.n_docs)
    descs = load_task_descriptions(a.n_desc)
    print(f"[data] {len(docs)} APS abstracts, {len(descs)} T2L task descriptions", flush=True)

    groups = {
        "aps_abstract": list(docs.text),
        "t2l_task_desc": [d for _, d in descs],
    }

    res = {"config": vars(a), "hypermod_dir": T["hypermod_dir"], "groups": {}}
    store = {}
    for name, texts in groups.items():
        e0 = embed_e0(T, texts).cpu()
        e1 = e0_to_e1(T, e0).cpu()
        sds = [e1_to_lora(T, e1[i]) for i in range(len(texts))]
        norms = [dw_norm(s) for s in sds]
        res["groups"][name] = {
            "n": len(texts),
            "E0_cos": cos_matrix_stats(e0.numpy()),
            "E1_cos": cos_matrix_stats(e1.numpy()),
            "E2_dw_cos": pairwise_dw_stats(sds),
            "dw_frob_norm": {"mean": float(np.mean(norms)), "sd": float(np.std(norms))},
        }
        store[name] = (texts, sds)
        g = res["groups"][name]
        print(f"\n[{name}] n={len(texts)}", flush=True)
        for k in ("E0_cos", "E1_cos", "E2_dw_cos"):
            s = g[k]
            print(f"  {k:10s} mean {s['mean']:+.4f} +- {s['sd']:.4f}   "
                  f"[p05 {s['p05']:+.4f}, p95 {s['p95']:+.4f}]", flush=True)
        print(f"  ||dW||_F   {g['dw_frob_norm']['mean']:.4f} +- {g['dw_frob_norm']['sd']:.4f}",
              flush=True)

    # ---- (B) endpoint decode -------------------------------------------------
    print("\n[endpoint decode]", flush=True)
    texts, sds = store["aps_abstract"]
    decodes = []
    for i in range(min(a.n_decode, len(sds))):
        lab = decode_lora(T, sds[i], LABEL_PROMPT, max_new_tokens=24)
        top = decode_lora(T, sds[i], TOPIC_PROMPT, max_new_tokens=60)
        decodes.append({"paper_id": int(docs.paper_id[i]), "title": docs.title[i],
                        "label_decode": lab, "topic_decode": top})
        print(f"  [{i}] {docs.title[i][:70]}\n      label: {lab}\n      topic: {top}", flush=True)
    res["endpoint_decodes"] = decodes

    # a task-description decode, for contrast (in-distribution input)
    dtexts, dsds = store["t2l_task_desc"]
    res["task_desc_decodes"] = []
    for i in range(min(3, len(dsds))):
        out = decode_lora(T, dsds[i], TOPIC_PROMPT, max_new_tokens=60)
        res["task_desc_decodes"].append({"task": descs[i][0], "topic_decode": out})
        print(f"  [desc/{descs[i][0]}] {out}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
