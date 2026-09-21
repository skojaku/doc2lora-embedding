"""Build the aligned cluster-label evaluation table (issue #24).

For every PACS node in the appendix hierarchy table (nodes.json: 4 fields, 8
divisions, 16 subdivisions) collect:
  - the ground-truth PACS label,
  - the decoded label from each method (D2L full-rank, vec2text, KeyLLM, ICAE),
  - one ANCESTOR (broader) and one DESCENDANT (narrower) node with real labels,
    sampled from the labelled PACS tree (groups.parquet) so Metric 2 can test
    whether a label sits at the right altitude.

The experiment renumbers fields (digit 3 = "Condensed matter", whose real PACS
divisions 75/71 live under PACS main 7), so field nodes are keyed by the
experiment digit while div/sub nodes are keyed by the real PACS code. We source
ancestors/descendants from real PACS codes for div/sub, and from nodes.json
children (+ root "Physics") for fields.

Output: label_eval_nodes.json
Run:    python label_eval_prep.py
"""
import json
import random
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
from bench_data import out_dir   # bench_data.py sits next to this file
DATA = out_dir("labels")           # where this chain writes

GROUPS = HERE.parent / "2026-05-28-concept-analogy-aps" / "results" / "groups.parquet"

# experiment field digit -> field name (combine_fields9.py NAMES, 4 used fields)
FIELD_NAME = {"0": "General", "2": "Classical physics",
              "3": "Condensed matter", "6": "Elementary particles"}
NAME_FIELD = {v: k for k, v in FIELD_NAME.items()}

# Every method is scored on what it NATIVELY produces -- no naming LLM is applied
# to any of them, so no method's label passes through a model that also judges it.
METHOD_FILES = {
    "doc2lora": "qwen_fullrank_field23.json",    # {code: label}    decoded in-model (Qwen3-4B)
    "vec2text": "vec2text_labels.json",          # {code: {label}}  raw GTR inversion (no prompt)
    "keyllm": "keyllm_faithful_labels.json",     # {code: kw-list}  raw KeyBERT/KeyLLM keywords
    "icae": "icae_raw.json",                     # {code: label}    raw ICAE decoder output
    "incontext": "incontext_labels.json",        # {code: label}    in-context text baseline (#29)
    # Text-to-LoRA (#151), one arm per embedding position in its pipeline: E0 the frozen
    # gte input, E1 the 64-d TaskEncoder output -- the only document-dependent learned
    # layer T2L has -- and E2 the generated LoRA factors, averaged as doc2lora averages
    # them. All three are RAW decoder output with no naming LLM, matching the native-
    # output protocol (00b77d2). workflow/scripts/t2l_label.py, CAP=2000
    # members per node (doc2lora's CAP, not ICAE's K=8).
    "t2l_gte": "t2l_gte_labels.json",            # {code: label}    E0
    "t2l_hidden": "t2l_hidden_labels.json",      # {code: label}    E1
    "t2l_dw": "t2l_dw_labels.json",              # {code: label}    E2
}

RNG = random.Random(42)


def load_method(fname):
    d = json.loads((HERE / fname).read_text())
    out = {}
    for k, v in d.items():
        s = v["label"] if isinstance(v, dict) else v
        # defensive: strip any <keywords> wrapper the KeyLLM backend leaves on the
        # raw keyword list and collapse newlines (no-op for the other methods).
        s = str(s or "").replace("<keywords>", " ").replace("</keywords>", " ")
        out[k] = " ".join(s.split())
    return out


def walk(node, parent, out):
    """Flatten nodes.json into records keyed by experiment code."""
    kind = node.get("kind")
    ref = node.get("ref")
    if kind == "field":
        code = NAME_FIELD[ref]
        gt = ref
    elif kind in ("div", "sub"):
        code = ref
        gt = node.get("gt")
    else:  # root
        for c in node.get("children", []):
            walk(c, node, out)
        return
    out[code] = {
        "code": code, "kind": kind, "gt_label": gt, "n": node.get("n"),
        "_children": [c for c in node.get("children", [])],
    }
    for c in node.get("children", []):
        walk(c, node, out)


def main():
    tree = json.loads((DATA / "nodes.json").read_text())
    recs = {}
    walk(tree, None, recs)

    methods = {m: load_method(f) for m, f in METHOD_FILES.items()}

    g = pd.read_parquet(GROUPS)
    by_code = {r.group_id: r for r in g.itertuples(index=False)}

    def glabel(code):
        r = by_code.get(code)
        return None if r is None else r.gt_label_text

    def gchildren(code, level):
        pid = f"{level}:{code}"
        return sorted(g[g.parent_id == pid].group_id.tolist())

    def norm(s):
        return " ".join(str(s or "").lower().split())

    ROOT_ANC = {"code": "root", "label": "Physics (the whole discipline)"}

    def climb_distinct(code, focal_gt):
        """Nearest real-PACS ancestor whose label differs from focal_gt; else root."""
        cur = by_code.get(code)
        seen = norm(focal_gt)
        while cur is not None and isinstance(cur.parent_id, str):
            ac = cur.parent_id.split(":")[-1]
            lab = glabel(ac)
            if lab and norm(lab) != seen:
                return {"code": ac, "label": lab}
            cur = by_code.get(ac)
        return dict(ROOT_ANC)

    def pick_distinct_child(children, focal_gt, anc_label):
        cand = [(c, glabel(c)) for c in children]
        cand = [(c, l) for c, l in cand if l and norm(l) != norm(focal_gt)
                and norm(l) != norm(anc_label)]
        if not cand:
            return None
        c, l = RNG.choice(cand)
        return {"code": c, "label": l}

    out = []
    for code, rec in recs.items():
        kind = rec["kind"]
        gt = rec["gt_label"]
        anc = desc = None
        if kind == "field":
            # ancestor = root "Physics"; descendant = a child division (nodes.json)
            # whose PACS label differs from the field name.
            anc = dict(ROOT_ANC)
            divs = sorted(c.get("ref") for c in rec["_children"] if c.get("kind") == "div")
            desc = pick_distinct_child(divs, gt, anc["label"])
        elif kind == "div":
            # ancestor = nearest distinct real-PACS main; descendant = a distinct
            # subdivision child.
            anc = climb_distinct(code, gt)
            desc = pick_distinct_child(gchildren(code, "division"), gt, anc["label"])
        elif kind == "sub":
            # ancestor = nearest distinct real-PACS division/main; specific-code
            # children are placeholder-labelled, so no usable descendant.
            anc = climb_distinct(code, gt)
            desc = None

        out.append({
            "code": code, "kind": kind, "gt_label": rec["gt_label"], "n": rec["n"],
            "ancestor": anc, "descendant": desc,
            "decoded": {m: methods[m].get(code) for m in METHOD_FILES},
        })

    out.sort(key=lambda r: (("field", "div", "sub").index(r["kind"]), r["code"]))
    (DATA / "label_eval_nodes.json").write_text(json.dumps(out, indent=2))
    # report
    miss = [(r["code"], m) for r in out for m in METHOD_FILES if not r["decoded"][m]]
    n2 = sum(bool(r["ancestor"]) and bool(r["descendant"]) for r in out)
    print(f"{len(out)} nodes | both-neighbour (2-sided Metric 2): {n2} | "
          f"ancestor-only: {sum(bool(r['ancestor']) and not r['descendant'] for r in out)}")
    if miss:
        print("MISSING decoded labels:", miss)
    for r in out[:6]:
        a = r["ancestor"]["label"] if r["ancestor"] else None
        d = r["descendant"]["label"] if r["descendant"] else None
        print(f"\n[{r['kind']} {r['code']}] gt={r['gt_label']!r}")
        print(f"   anc={a!r}\n   desc={d!r}")
        print(f"   doc2lora={r['decoded']['doc2lora']!r}  keyllm={r['decoded']['keyllm']!r}")
    print(f"\nwrote {HERE / 'label_eval_nodes.json'}")


if __name__ == "__main__":
    main()
