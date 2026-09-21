"""Does the T2L pipeline work at all, on the inputs T2L was built for? (issue #151)

This is the control that decides how `sanity_degeneracy.py`'s decode failure may be
written up. It is NOT a comparison arm and never enters a results table -- per the
issue's Decision, no arm gets a bespoke input. Here we deliberately feed T2L the
in-distribution input (one of its own task descriptions, verbatim from
`tasks/<task>/metadata.yaml`) to show the implementation reproduces the paper's
behaviour.

  adapter OFF (zeroed)   vs   adapter ON (generated from the task description)

Task choice matters. The first attempt used GSM8K and the adapter LOOKED harmful
(0.463 -> 0.163), but `trained_t2l/mistral_7b_t2l/args.yaml` lists 479 training tasks
and **gsm8k is not among them** -- it is a held-out generalization task, and the
adapter's effect there is to suppress chain-of-thought (the LoL tasks it was trained
on have short direct answers), which is exactly what GSM8K needs. A pipeline check
has to use a task the hypernetwork was TRAINED on, so the default here is `lol_636`,
which appears in `train_ds_names`.

Input/answer formatting is T2L's own: `get_preprocessing_fn`'s `lol_` branch carves
`task_def` / `problem` / `answer` out of the raw `input`/`output` columns, and
`user_prompt_template` ("{task_def}\n\n{problem}") wraps them.

Note on contamination: LoL tasks are small and T2L trained on `train[:10000]`, which
for most of them is the whole split, so the eval rows may be training rows. That is
acceptable *for a plumbing check* -- the question is whether the adapter moves
behaviour in the direction its training intended, not how well it generalizes -- and
`--offset` shifts to unseen rows for the tasks that are large enough.

If ON > OFF, our loading/injection is faithful and the failure to decode documents is
a property of T2L, not of our wiring. If ON <= OFF, we have a setup bug and nothing
else in this experiment may be reported.

Run:
  export PYTHONPATH=$T2L_SRC
  CUDA_VISIBLE_DEVICES=0 python task_validity.py [--task lol_636] [--n 80]
"""
import argparse
import os
import glob
import json
import re
import sys
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2l_common import (  # noqa: E402
    ROOT, T2L_SRC, HYPERMOD_DIR, load_t2l, embed_e0, e0_to_e1, e1_to_lora, apply_lora, build_prompt,
)

T2L_REPO = Path(os.environ.get("T2L_REPO", str(Path(T2L_SRC).parent)))


def lol_preprocess(example):
    """T2L's `get_preprocessing_fn` lol_ branch, verbatim."""
    txt = example["input"]
    task_def = txt.split("Definition: ")[1].split("\n\nPositive Example")[0]
    task_def += " Please complete the task without any explanation."
    if len(example["output"]) > 1:
        task_def += "\nThe answer should be a comma-separated list of possible completions."
    problem = txt.split("Now complete the following example -")[1].split("Input: ")[1].split("\nOutput:")[0]
    return dict(task_def=task_def, problem=problem, answer=", ".join(example["output"]))


def norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().strip(".").lower()


def pick_description(task, meta):
    """Prefer a HELD-OUT description from the checkpoint's own eval config."""
    p = glob.glob(f"{HYPERMOD_DIR}/args.yaml")
    if p:
        info = (yaml.safe_load(open(p[0])).get("eval_ds_info") or {}).get(task)
        if info and info.get("descriptions"):
            return info["descriptions"][0], "held-out (args.yaml eval_ds_info)"
    return meta["descriptions"][0], "metadata.yaml descriptions[0]"


@torch.no_grad()
def answer(T, fields, meta, max_new_tokens=96):
    """Ask one item in EXACTLY T2L's training-time format (metadata-driven)."""
    tok, model = T["tokenizer"], T["model"]
    text = build_prompt(
        T,
        meta["user_prompt_template"].format(**fields),
        system_message=(meta.get("system_message") or "").format(**fields),
        assistant_prefill=(meta.get("assistant_prefill") or "").format(**fields),
    )
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    out = model.generate(input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="lol_636", help="must be in args.yaml train_ds_names")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--offset", type=int, default=0, help="skip N rows (to dodge training rows)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_path = a.out or str(HERE / f"results/task_validity_{a.task}.json")

    from datasets import load_dataset

    meta = yaml.safe_load((T2L_REPO / "tasks" / a.task / "metadata.yaml").read_text())
    args_y = yaml.safe_load(open(glob.glob(f"{HYPERMOD_DIR}/args.yaml")[0]))
    in_train = a.task in (args_y.get("train_ds_names") or [])
    desc, desc_src = pick_description(a.task, meta)
    print(f"[task] {a.task}  in_train_ds_names={in_train}")
    print(f"[desc/{desc_src}] {desc}")
    print(f"[user_tpl] {meta['user_prompt_template']!r}", flush=True)
    if not in_train:
        print("WARNING: task is NOT in the hypernet's training set -- this is a "
              "generalization test, not a pipeline check.", flush=True)

    kw = dict(meta["ds_kwargs"])
    ds = load_dataset(kw["path"], kw.get("name", "default"), split="train")
    rows = [lol_preprocess(ds[i]) for i in range(a.offset, min(a.offset + a.n, len(ds)))]
    print(f"[data] {len(rows)} rows (offset {a.offset} of {len(ds)})", flush=True)

    T = load_t2l()
    sd_on = e1_to_lora(T, e0_to_e1(T, embed_e0(T, [desc]))[0])
    sd_off = {k: torch.zeros_like(v) for k, v in sd_on.items()}

    res = {"task": a.task, "in_train_ds_names": in_train, "n": len(rows),
           "description": desc, "description_source": desc_src,
           "user_prompt_template": meta["user_prompt_template"], "conditions": {}}
    for cond, sd in (("adapter_off", sd_off), ("adapter_on", sd_on)):
        apply_lora(T, sd)
        hits, recs = 0, []
        for i, r in enumerate(rows):
            txt = answer(T, r, meta)
            ok = norm(txt) == norm(r["answer"])
            hits += ok
            recs.append({"pred": txt, "gold": r["answer"], "ok": bool(ok)})
            if i < 2:
                print(f"  [{cond}/{i}] gold={r['answer']!r}\n            pred={txt[:160]!r}", flush=True)
        acc = hits / len(rows)
        res["conditions"][cond] = {"accuracy": acc, "hits": hits, "records": recs}
        print(f"[{cond}] exact-match {hits}/{len(rows)} = {acc:.3f}", flush=True)

    d = res["conditions"]["adapter_on"]["accuracy"] - res["conditions"]["adapter_off"]["accuracy"]
    res["delta"] = d
    print(f"\n[verdict] adapter_on - adapter_off = {d:+.3f}  "
          f"({'pipeline faithful' if d > 0 else 'SETUP SUSPECT -- do not report other arms'})")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(res, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
