"""Text-to-LoRA labels for the PACS node comparison (issue #151).

One `{code: label}` JSON per embedding position, ready to register in
`exps/2026-06-11-baseline-trees/label_eval_prep.py:METHOD_FILES`:

    t2l_gte_labels.json     E0 - mean of the frozen gte-large vectors, then the whole
                                 hypernetwork runs on the mean
    t2l_hidden_labels.json  E1 - mean of the TaskEncoder outputs (64-d), then the rest
                                 of the hypernetwork runs on the mean
    t2l_dw_labels.json      E2 - mean of the generated LoRA factors (A and B averaged
                                 separately, rank stays 8), which is the structural
                                 analogue of what doc2lora does

Nodes and membership are taken from `paper_groups.parquet` exactly as
`compute_qwen_fullrank_means.py` does, and up to CAP members are sampled per node with
the same seed, so the arms stay paired. **CAP defaults to 2,000, matching doc2lora --
not ICAE's K=8.** gte is cheap, so there is no reason to run this baseline on eight
documents when the method it is compared against averages two thousand; running it weak
is how a baseline gets dismissed as not really run (survey pattern P2).

The decode prompt is the shared one every arm uses
(`exps/2026-06-11-baseline-trees/shared_prompt.py`), delivered in T2L's own training
prompt format (`t2l_common.build_prompt`).

Per-document adapters are never stored (6.5 MiB each, 56k documents would be 364 GB);
`LoraAccumulator` streams the mean. Results are saved per node, so the job resumes.

  export PYTHONPATH=$T2L_SRC
  CUDA_VISIBLE_DEVICES=0 python t2l_label.py [--cap 2000]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2l_common import (  # noqa: E402
    ROOT, load_t2l, embed_e0, e0_to_e1, e1_to_lora, LoraAccumulator, decode_lora,
)

sys.path.insert(0, str(ROOT / "exps/2026-06-11-baseline-trees"))
from shared_prompt import LABEL_PROMPT  # noqa: E402

CA = ROOT / "exps/2026-05-28-concept-analogy-aps"
SEED = 0
MAX_NEW = 24

# identical to compute_qwen_fullrank_means.py
SPEC = [
    {"fid": 3, "divs": [("75", ["75.10", "75.30"]), ("71", ["71.10", "71.20"])]},
    {"fid": 0, "divs": [("03", ["03.67", "03.65"]), ("05", ["05.45", "05.40"])]},
    {"fid": 6, "divs": [("11", ["11.15", "11.10"]), ("12", ["12.38", "12.60"])]},
    {"fid": 2, "divs": [("42", ["42.50", "42.65"]), ("47", ["47.27", "47.20"])]},
]


def node_list():
    nodes = []
    for f in SPEC:
        nodes.append((str(f["fid"]), "main", f["fid"]))
        for code, subs in f["divs"]:
            nodes.append((code, "division", code))
            for s in subs:
                nodes.append((s, "subdivision", s))
    return nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=2000, help="members averaged per node")
    ap.add_argument("--batch-size", type=int, default=16, help="gte encode batch")
    ap.add_argument("--state", default=str(HERE / "results/t2l_label_state.json"))
    a = ap.parse_args()

    pg = pd.read_parquet(CA / "results/paper_groups.parquet")
    txt = pd.read_parquet(ROOT / "data/aps/paper_text_pid.parquet",
                          columns=["paper_id", "title", "abstract"])
    txt = txt[txt.abstract.notna()]
    txt = txt.assign(title=txt.title.fillna("").str.strip())
    id2text = dict(zip(txt.paper_id.astype(int),
                       (txt.title + ". " + txt.abstract.str.strip()).str.strip(". ")))
    rng = np.random.default_rng(SEED)

    def members(level, val):
        if level == "main":
            m = pg[pg.main_class_id == val].paper_id.values
        elif level == "division":
            m = pg[pg.division.astype(str) == val].paper_id.values
        else:
            m = pg[pg.subdivision.astype(str) == val].paper_id.values
        return np.array([int(p) for p in m if int(p) in id2text])

    state_path = Path(a.state)
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    nodes = node_list()
    todo = [n for n in nodes if n[0] not in state]
    print(f"[nodes] {len(nodes)} total, {len(todo)} to do", flush=True)
    if not todo:
        print("nothing to do")

    T = load_t2l() if todo else None

    for code, level, val in todo:
        ids = members(level, val)
        if len(ids) > a.cap:
            ids = rng.choice(ids, a.cap, replace=False)
        texts = [id2text[int(p)] for p in ids]

        e0 = embed_e0(T, texts, batch_size=a.batch_size)          # [n, 1024]
        e1 = e0_to_e1(T, e0)                                      # [n, 64]
        acc = LoraAccumulator()
        for i in range(len(e1)):
            acc.add(e1_to_lora(T, e1[i]))

        # E0: average in gte space, then run the WHOLE hypernetwork on the mean
        lab_gte = decode_lora(T, e1_to_lora(T, e0_to_e1(T, e0.mean(0, keepdim=True))[0]),
                              LABEL_PROMPT, max_new_tokens=MAX_NEW)
        # E1: average the TaskEncoder outputs, then run the rest of the hypernetwork
        lab_hidden = decode_lora(T, e1_to_lora(T, e1.mean(0)), LABEL_PROMPT,
                                 max_new_tokens=MAX_NEW)
        # E2: average the generated factors (A and B separately; rank stays 8)
        lab_dw = decode_lora(T, acc.mean(), LABEL_PROMPT, max_new_tokens=MAX_NEW)

        state[code] = {"level": level, "n_used": int(len(texts)),
                       "t2l_gte": lab_gte, "t2l_hidden": lab_hidden, "t2l_dw": lab_dw}
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2))
        print(f"  [{code:<6} {level:<11} n={len(texts):>5}] "
              f"gte={lab_gte!r} | hidden={lab_hidden!r} | dw={lab_dw!r}", flush=True)

    for key, fname in (("t2l_gte", "t2l_gte_labels.json"),
                       ("t2l_hidden", "t2l_hidden_labels.json"),
                       ("t2l_dw", "t2l_dw_labels.json")):
        out = {c: v[key] for c, v in state.items()}
        (HERE / fname).write_text(json.dumps(out, indent=2))
        print(f"wrote {HERE / fname} ({len(out)} nodes)")


if __name__ == "__main__":
    main()
