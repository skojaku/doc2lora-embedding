"""Shared machinery for the comprehensive simplex-fidelity benchmark (#60).

One corner-set spec format is used end to end. A spec lives in
``corners_<set>.json`` (in THIS folder, kept self-contained) with::

    {"set": "<name>", "keys": ["A", "B", "C"],
     "prompt": "<optional decode-prompt override>",
     "corners": {"A": {"name": "...", "lead": "<anchor document>",
                       "words": ["...", ...]}, "B": {...}, "C": {...}}}

The decode drivers (``decode_<channel>.py``) turn a spec + barycentric grid into
``results/sentence_<set>_<channel>.json``; the OFF-TOPIC-aware LLM judge
(``judge_offtopic.py``) tags every decoded passage's concepts as A/B/C/OTHER into
``results/judge_<set>_<channel>.json``; ``compute_metrics.py`` aggregates
Kendall-tau + fidelity per channel across all sets.

Off-topic is the key #60 addition: the original A/B/C-only judge could not flag
hallucinated content (it simply left it untagged), so fidelity was vacuously
~1.0 for every method. Here every concept the judge sees is forced into exactly
one of A/B/C/OTHER, so off-topic mass is measured rather than dropped.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
from bench_data import out_dir   # bench_data.py sits next to this file
RESULTS = str(out_dir("pair_axis"))   # where the pair-axis chain writes

GRID_DEN = 12                       # corners, edge midpoints, centroid all on-grid
POLES = ["A", "B", "C"]

# Few-sentence decode prompt (neutral, NOT feature-naming -- a leading prompt
# collapses every grid point to one prior; see exps/2026-06-12-idea-simplex notes).
SENT_PROMPT = ("Write three or four sentences describing the research topic and "
               "the central concepts it combines.")

# Cross-lab judge roster (OpenRouter slugs). gemini-3.5-flash dropped from the
# #33/#24 5-model pool on cost grounds: at $1.5/$9 per 1M in/out tokens it was
# ~$5.5 of an $8.6 full sweep for one of five votes; the 4-judge majority is
# robust to its removal. Re-add the line below for the full 5-judge protocol.
# Modern cheap+strong 5-judge roster (Jun 2026), one per provider for independence.
JUDGE_MODELS = {
    "minimax-m3": "minimax/minimax-m3",
    "deepseek-v4-flash": "deepseek/deepseek-v4-flash",
    "grok-4.3": "x-ai/grok-4.3",
    "gemini-3.1-flash-lite": "google/gemini-3.1-flash-lite",
    "qwen3.7-plus": "qwen/qwen3.7-plus",
}


def barycentric_grid(den=GRID_DEN):
    """All (i, j, k) with i+j+k=den, i,j,k>=0 -> integer triples in grid order."""
    pts = []
    for i in range(den + 1):
        for j in range(den - i + 1):
            pts.append((i, j, den - i - j))
    return pts


def load_corners(path):
    """Load a corner spec -> dict(keys, names, leads, words, prompt, set)."""
    w = json.load(open(path))
    keys = w["keys"]
    return {
        "set": w.get("set") or w.get("name") or os.path.basename(path),
        "keys": keys,
        "names": {k: w["corners"][k]["name"] for k in keys},
        "leads": {k: w["corners"][k]["lead"] for k in keys},
        "words": {k: w["corners"][k].get("words", []) for k in keys},
        "prompt": w.get("prompt"),
    }


def corners_path(set_name):
    return os.path.join(HERE, f"corners_{set_name}.json")


def sentence_path(set_name, channel):
    return os.path.join(RESULTS, f"sentence_{set_name}_{channel}.json")


def judge_path(set_name, channel):
    return os.path.join(RESULTS, f"judge_{set_name}_{channel}.json")


def write_sentences(set_name, channel, spec, samples, encoder):
    """Persist a decoded grid in the one schema every judge/metric step reads."""
    os.makedirs(RESULTS, exist_ok=True)
    out = sentence_path(set_name, channel)
    json.dump({"set": set_name, "channel": channel, "encoder": encoder,
               "keys": spec["keys"], "names": spec["names"],
               "decode_prompt": spec.get("prompt") or SENT_PROMPT,
               "samples": samples}, open(out, "w"), indent=2, ensure_ascii=False)
    return out


# ── Off-topic-aware LLM judge prompt ────────────────────────────────────────
def build_judge_prompt(passage, spec):
    """Force every concept in the passage into exactly one of A/B/C/OTHER.

    OTHER is the #60 fidelity hook: content attributable to none of the three
    corners (hallucinated / generic / off-topic) is counted, not silently
    dropped as the A/B/C-only tagger did.
    """
    lines = ["Three reference concepts:"]
    for p in spec["keys"]:
        name = spec["names"][p]
        terms = ", ".join(spec["words"][p][:14])
        lead = spec["leads"][p].strip().replace("\n", " ")
        lines.append(f"{p} = {name} — {lead[:300]}\n    Characteristic terms: {terms}.")
    lines += [
        "",
        "Passage to analyse:",
        f'"""{passage.strip()}"""',
        "",
        "Break the passage into its distinct substantive concepts/claims. For "
        "EACH concept, decide which single reference it belongs to: tag it A, B, "
        "or C if it genuinely concerns that concept, or OTHER if it concerns none "
        "of the three (generic filler, hallucinated, or about an unrelated topic). "
        "Judge by substance, not incidental word overlap. Every concept gets "
        "exactly one tag. List 2 to 8 concepts.",
        "",
        "Reason in AT MOST 30 words, then respond with ONLY a JSON object:",
        '{"reasoning": "<=30 words", '
        '"concepts": [{"concept": "<short phrase>", "tag": "A"|"B"|"C"|"OTHER"}, ...]}',
    ]
    return "\n".join(lines)


def parse_judge(txt):
    """Return list of {'concept','tag'} from a judge reply; robust to truncation.

    Returns None on unparseable output (judge abstains for that passage).
    """
    objs = re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", txt, flags=re.S)
    # The outermost object is the whole reply; find the one carrying "concepts".
    for cand in re.findall(r"\{.*\}", txt, flags=re.S):
        try:
            o = json.loads(cand)
        except Exception:
            continue
        if isinstance(o, dict) and "concepts" in o:
            return _norm_concepts(o["concepts"])
    # salvage individual {concept,tag} objects from partial JSON
    salv = []
    for m in re.finditer(r'"concept"\s*:\s*"([^"]*)"\s*,\s*"tag"\s*:\s*"?([A-Za-z]+)"?', txt):
        salv.append({"concept": m.group(1), "tag": m.group(2)})
    return _norm_concepts(salv) if salv else None


def _norm_concepts(seq):
    out = []
    for c in seq or []:
        if not isinstance(c, dict):
            continue
        tag = str(c.get("tag", "")).strip().upper()[:5]
        tag = tag if tag in ("A", "B", "C") else "OTHER"
        out.append({"concept": str(c.get("concept", ""))[:200], "tag": tag})
    return out
