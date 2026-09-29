"""In-context TEXT baseline using Doc2LoRA's OWN Qwen generator (issues #24, #29).

The cleanest channel control. Doc2LoRA's field labels are produced by decoding the
averaged idea-gene with the Qwen3-4B doc2lora model under FIELD23_PROMPT
(decode_fullrank_field23.py). This baseline loads the SAME model and asks the SAME
prompt, but feeds the cluster's actual papers as text instead of internalizing a
gene -- so the only difference from Doc2LoRA is the channel (read-the-text vs.
decode-the-averaged-gene), not the model OR the instruction. It also avoids the
mild circularity of an OpenRouter reader (minimax-m3) that is also a judge.

Protocol (issue #29 budget): the DOC_BUDGET=20 members nearest the cluster centroid
(nodes.json, nearest-first), title + abstract each; greedy decode, max 16 tokens.

Reads:  nodes.json
Writes: incontext_labels.json   {code: label}
Run (GPU): CUDA_VISIBLE_DEVICES=0 python incontext_label_qwen.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # repository root, not a fixed ~/projects path
HERE = Path(__file__).resolve().parent
from bench_data import out_dir   # bench_data.py sits next to this file
DATA = out_dir("labels")           # where this chain writes

for _p in ("libs/legacy", "libs/doc2lora"):   # doc2lora_legacy lives in libs/legacy
    sys.path.insert(0, str(ROOT / _p))
from doc2lora_legacy import load_model, generate_text  # noqa: E402

CKPT = ROOT / "data/agent_assets/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"
NODES = DATA / "nodes.json"
OUT = DATA / "incontext_labels.json"

DOC_BUDGET = 20
ABS_CHARS = 600

# Verbatim shared field prompt -- identical to decode_fullrank_field23.FIELD23_PROMPT.
FIELD23_PROMPT = (
    "In 2 to 3 words, name the scientific field that all of these documents "
    "belong to. Reply with only the field name."
)


def build_prompt(docs):
    parts = []
    for i, d in enumerate(docs[:DOC_BUDGET], 1):
        title = (d.get("title") or "").strip()
        ab = (d.get("abstract") or "").replace("\n", " ").strip()[:ABS_CHARS]
        parts.append(f"[{i}] {title}\n{ab}")
    n = min(len(docs), DOC_BUDGET)
    return (f"Here are {n} research-paper abstracts from one cluster:\n\n"
            + "\n\n".join(parts) + f"\n\n{FIELD23_PROMPT}")


def normalize_label(text):
    lines = [l.strip(" \t\"'*`-") for l in text.splitlines() if l.strip()]
    lab = lines[0] if lines else text.strip()
    lab = re.sub(r"^(cluster label|cluster|label|topic|field)\s*[:\-]\s*", "",
                 lab, flags=re.I)
    return lab.strip(" \t\"'*`.")


def nodes_with_codes(tree):
    """field -> experiment digit (fid); division/subdivision -> PACS code (ref)."""
    out = []
    for f in tree["children"]:
        out.append((str(f["fid"]), f["docs"]))
        for d in f["children"]:
            out.append((d["ref"], d["docs"]))
            for s in d["children"]:
                out.append((s["ref"], s["docs"]))
    return out


def main():
    tree = json.loads(NODES.read_text())
    model, gen_tok, _ = load_model(str(CKPT), mode="full")
    out = {}
    for code, docs in nodes_with_codes(tree):
        if not docs:
            continue
        model.reset()
        txt = generate_text(model, gen_tok, build_prompt(docs), max_new_tokens=16)
        out[code] = normalize_label(txt)
        model.reset()
        print(f"  {code:>6} | {out[code]}", flush=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {OUT.name} ({len(out)} nodes, reader=qwen_4b_d2l, budget={DOC_BUDGET})")


if __name__ == "__main__":
    main()
