"""Stronger in-context TEXT baseline for cluster labeling (issue #29).

The "what if you just let a text LLM READ the real papers?" reference. Unlike
KeyLLM / ICAE / vec2text -- which emit a lossy artifact (keywords / a
reconstruction / a GTR inversion) that a separate model then has to name -- this
baseline hands the reader the actual cluster documents and asks the SAME 2-3 word
field question the Doc2LoRA decoder is asked. It therefore needs no separate
naming step: the reader emits the scoreable label directly.

Protocol (issue #29 "standardize the document budget"):
  - Input: the DOC_BUDGET=20 members nearest the cluster centroid (the same fixed
    budget KeyLLM is held to; nodes.json lists members nearest-first), title +
    abstract each.
  - Prompt: verbatim FIELD23_PROMPT, identical to decode_fullrank_field23.py so
    the only thing that differs from Doc2LoRA is the channel (read-the-text vs.
    decode-the-averaged-gene), not the instruction.
  - Reader: minimax-m3 (OpenRouter), matching the Problem-B naming model.

If Doc2LoRA matches an LLM that actually reads the papers, the labeling claim is
much stronger. Responses are cached on disk so re-runs are free.

Reads:  nodes.json            (centroid-nearest members per PACS node)
Writes: incontext_labels.json  {code: label}
Run:    set -a; . ../../.env; set +a; python incontext_label.py
"""
import hashlib
import json
import os
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
NODES = HERE / "nodes.json"
OUT = HERE / "incontext_labels.json"
CACHE = HERE / "incontext_cache"

OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")
URL = "https://openrouter.ai/api/v1/chat/completions"
READER = "minimax/minimax-m3"
DOC_BUDGET = 20  # members nearest the centroid the reader sees (== KeyLLM budget)

# Verbatim shared field prompt -- identical to decode_fullrank_field23.FIELD23_PROMPT.
FIELD23_PROMPT = (
    "In 2 to 3 words, name the scientific field that all of these documents "
    "belong to. Reply with only the field name."
)
SYS = (
    "You are a research librarian. You read a set of paper abstracts from a single "
    "cluster and name the one scientific field they share, as concisely as possible."
)


def build_user(docs):
    parts = []
    for i, d in enumerate(docs[:DOC_BUDGET], 1):
        title = (d.get("title") or "").strip()
        abstract = (d.get("abstract") or "").strip()
        parts.append(f"[{i}] {title}\n{abstract}")
    body = "\n\n".join(parts)
    n = min(len(docs), DOC_BUDGET)
    return (f"Here are {n} research-paper abstracts from one cluster:\n\n{body}\n\n"
            f"{FIELD23_PROMPT}")


def ask(system, user, retries=4):
    # minimax-m3 is a reasoning model: give it ample headroom so the final answer
    # is not starved by the reasoning trace (a 256-token cap left `content` empty).
    CACHE.mkdir(exist_ok=True)
    ck = CACHE / (hashlib.sha1(f"{READER}\x00{system}\x00{user}".encode()).hexdigest() + ".json")
    if ck.exists():
        cached = json.loads(ck.read_text())["text"]
        if cached.strip():
            return cached
    body = {"model": READER,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0, "max_tokens": 2000}
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {OR_KEY}"},
                              json=body, timeout=180)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            txt = (msg.get("content") or "").strip()
            if not txt:  # budget spent on reasoning -> salvage its concluding line
                reasoning = (msg.get("reasoning") or "").strip()
                txt = reasoning.splitlines()[-1].strip() if reasoning else ""
            if txt:
                ck.write_text(json.dumps({"text": txt}))
                return txt
            last = "empty content and reasoning"
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"OpenRouter call failed for {READER}: {last}")


def clean(label):
    label = (label or "").strip()
    if "</think>" in label:  # reasoning models occasionally leak a think block
        label = label.split("</think>")[-1].strip()
    lines = [ln.strip() for ln in label.splitlines() if ln.strip()]
    label = lines[-1] if lines else label
    label = label.strip().strip('"').strip("'").strip()
    n = len(label)  # collapse an exact doubled phrase ("Quantum opticsQuantum optics")
    if n and n % 2 == 0 and label[:n // 2] == label[n // 2:]:
        label = label[:n // 2]
    return label.strip()


def nodes_with_codes(tree):
    """(code, docs) per scoreable node, keyed exactly as the other method files:
    field -> experiment digit (str fid), division/subdivision -> PACS code (ref).
    The root is not scored, so it is skipped."""
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
    out = {}
    for code, docs in nodes_with_codes(tree):
        if not docs:
            continue
        label = clean(ask(SYS, build_user(docs)))
        out[code] = label
        print(f"  {code:>6} | {label}", flush=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {OUT.name} ({len(out)} nodes, reader={READER}, budget={DOC_BUDGET})")


if __name__ == "__main__":
    main()
