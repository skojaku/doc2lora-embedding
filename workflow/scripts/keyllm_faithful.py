"""Faithful KeyLLM baseline with a minimax-m3 backend (issue #24).

Runs the real KeyBERT/KeyLLM library (Grootendorst), exactly like the original
keyllm_label.py, but swaps the ollama glm-5.1 backend for minimax-m3 via
OpenRouter's OpenAI-compatible endpoint. Each PACS node is treated as a cluster;
KeyLLM is run on the node's MEDOID document (docs[0], the nearest-centroid member;
title + abstract up to 1200 chars) with the default KeyLLM prompt. The native
output is a KEYWORD LIST -- we do NOT add a separate naming step, so this is as
faithful to KeyLLM as possible (the keyword list IS the label).

Nodes are keyed exactly like the other method files: field -> experiment digit
(fid), division/subdivision -> PACS code (ref).

Output: keyllm_faithful_labels.json   {code: "kw1, kw2, ..."}
Run:    set -a; . ../../.env; set +a; python keyllm_faithful.py
"""
import json
import os
from pathlib import Path

import openai
from keybert import KeyLLM
from keybert.llm import OpenAI as KBOpenAI

HERE = Path(__file__).resolve().parent
from bench_data import out_dir   # bench_data.py sits next to this file
DATA = out_dir("labels")           # where this chain writes

NODES = DATA / "nodes.json"
OUT = DATA / "keyllm_faithful_labels.json"
MODEL = "minimax/minimax-m3"
ABS_CHARS = 1200

client = openai.OpenAI(base_url="https://openrouter.ai/api/v1",
                       api_key=os.environ.get("OPENROUTER_API_KEY", ""))
# default KeyLLM chat prompt (prompt=None); ample max_tokens so the reasoning
# model is not starved, temperature 0 for reproducibility.
llm = KBOpenAI(client, model=MODEL, chat=True,
               generator_kwargs={"temperature": 0, "max_tokens": 512})
kw_model = KeyLLM(llm)


def representative(node):
    docs = node.get("docs", [])
    if not docs:
        return ""
    d = docs[0]
    rep = (d.get("title") or "").strip()
    a = (d.get("abstract") or "")[:ABS_CHARS].strip()
    return (rep + ("\n" + a if a else "")).strip()


def label_node(node):
    rep = representative(node)
    if not rep:
        return ""
    kws = kw_model.extract_keywords([rep])      # vanilla KeyLLM, default prompt
    flat = kws[0] if kws and isinstance(kws[0], list) else kws
    raw = ", ".join(k for k in flat if k)
    # minimax-m3 wraps its list in <keywords>...</keywords>; strip the tags and
    # collapse newlines so the keyword string matches the original 80-char form.
    raw = raw.replace("<keywords>", " ").replace("</keywords>", " ")
    return " ".join(raw.split())[:80]


def nodes_with_codes(tree):
    out = []
    for f in tree["children"]:
        out.append((str(f["fid"]), f))
        for d in f["children"]:
            out.append((d["ref"], d))
            for s in d["children"]:
                out.append((s["ref"], s))
    return out


def main():
    tree = json.loads(NODES.read_text())
    out = {}
    for code, node in nodes_with_codes(tree):
        out[code] = label_node(node)
        print(f"  {code:>6} | {out[code]}", flush=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {OUT.name} ({len(out)} nodes, backend={MODEL})")


if __name__ == "__main__":
    main()
