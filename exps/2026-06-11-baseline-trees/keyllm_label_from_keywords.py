"""Turn KeyLLM keyword lists into concise cluster labels (Problem-B naming step).

KeyLLM natively emits a keyword list (it is a keyword extractor). To make it
comparable to the other methods that emit a named topic, we feed each node's
KeyLLM keyword list to the naming model and ask for a 2-3 word field name -- the
same output budget as Doc2LoRA and the in-context baseline (issue #24/#29). The
namer sees ONLY the keywords, never the papers.

The whole KeyLLM path is now minimax-m3 (issue #24): the keyword lists come from
keyllm_faithful.py (real KeyBERT/KeyLLM on the medoid document, minimax-m3
backend) -- fresh, not the frozen/truncated glm-5.1 lists -- and the naming step
below also uses minimax-m3.

Reads:  keyllm_faithful_labels.json   {node: "kw1, kw2, ..."}
Output: keyllm_label_from_keywords.json  {node: label}

Run: set -a; . ../../.env; set +a; python keyllm_label_from_keywords.py
"""
import json
import os
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "minimax/minimax-m3"
KEYWORDS_FILE = HERE / "keyllm_faithful_labels.json"

PROMPT = (
    # Same 2-3-word field budget as decode_fullrank_field23.FIELD23_PROMPT, so the
    # KeyLLM naming step is held to the same output constraint as Doc2LoRA and the
    # in-context baseline (otherwise KeyLLM's looser budget yields more specific
    # labels and an unfair edge on the label metrics -- issue #24/#29).
    "The following keywords were extracted from a cluster of physics research "
    "papers:\n\n{kw}\n\nIn 2 to 3 words, name the scientific field that this "
    "cluster belongs to. Reply with only the field name."
)


def ask(prompt):
    # Reasoning model: ample max_tokens so the answer is not starved by reasoning
    # (which would otherwise leave content=None); salvage the reasoning tail if so.
    body = {"model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2, "max_tokens": 2000}
    r = requests.post(URL, headers={"Authorization": f"Bearer {OR_KEY}"},
                      json=body, timeout=180)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    txt = (msg.get("content") or "").strip()
    if not txt:
        reasoning = (msg.get("reasoning") or "").strip()
        txt = reasoning.splitlines()[-1].strip() if reasoning else ""
    return txt


def clean(label):
    label = (label or "").strip().strip('"').strip("'").strip()
    # drop any stray think tags / leading bullet
    if "</think>" in label:
        label = label.split("</think>")[-1].strip()
    label = label.splitlines()[-1].strip() if label else label
    n = len(label)  # collapse an exact doubled phrase
    if n and n % 2 == 0 and label[:n // 2] == label[n // 2:]:
        label = label[:n // 2]
    return label


def main():
    if not KEYWORDS_FILE.exists():
        sys.exit(f"missing {KEYWORDS_FILE.name}; run keyllm_faithful.py first")
    keywords = json.loads(KEYWORDS_FILE.read_text())
    out = {}
    for code, kw in keywords.items():
        label = clean(ask(PROMPT.format(kw=kw)))
        out[code] = label
        print(f"  {code:>6} | {label}")
    (HERE / "keyllm_label_from_keywords.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False)
    )
    print("wrote keyllm_label_from_keywords.json")


if __name__ == "__main__":
    main()
