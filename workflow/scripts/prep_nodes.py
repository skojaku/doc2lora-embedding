"""Shared prep for the ICAE / RAG comparison trees.

For the SAME node set as the doc2lora decoded tree (4 fields, 2 divisions each,
2 subfields each, + whole-corpus root), retrieve the member documents nearest
the cluster centroid in SBERT (all-mpnet) space and dump their title+abstract.
Both baselines consume this identical retrieval so the only thing that differs
between figures is the labeling method.

Output: nodes.json  (nested tree mirroring the figure structure, with a `docs`
list of the top-K centroid-nearest members per node).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]   # repository root, not a fixed ~/projects path
RES = ROOT / "data/pacs/results"
TOPK = 30          # nearest-centroid members kept per node
ROOT_SAMPLE = 6000  # random members used to estimate the corpus centroid

# field cluster-id -> divisions -> subfields  (identical to the doc2lora figure)
SPEC = [
    {"fid": 3, "divs": [{"c": "75", "subs": ["75.10", "75.30"]},
                        {"c": "71", "subs": ["71.10", "71.20"]}]},
    {"fid": 0, "divs": [{"c": "03", "subs": ["03.67", "03.65"]},
                        {"c": "05", "subs": ["05.45", "05.40"]}]},
    {"fid": 6, "divs": [{"c": "11", "subs": ["11.15", "11.10"]},
                        {"c": "12", "subs": ["12.38", "12.60"]}]},
    {"fid": 2, "divs": [{"c": "42", "subs": ["42.50", "42.65"]},
                        {"c": "47", "subs": ["47.27", "47.20"]}]},
]


def main():
    pg = pd.read_parquet(RES / "paper_groups.parquet")
    gl = pd.read_parquet(RES / "groups.parquet")          # gt labels + counts
    txt = pd.read_parquet(ROOT / "data/aps/paper_text_pid.parquet",
                          columns=["paper_id", "title", "abstract", "text"])
    z = np.load(ROOT / "data/aps/embeddings/sbert_allmpnet.npz")
    pids, vecs = z["paper_ids"].astype(np.int64), z["vecs"].astype(np.float32)
    vn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    pid2row = {int(p): i for i, p in enumerate(pids)}
    txt = txt.set_index("paper_id")

    def gt_of(level, gid):
        r = gl[(gl.level == level) & (gl.group_id.astype(str) == str(gid))]
        return (r.iloc[0].gt_label_text, int(r.iloc[0].n_papers)) if len(r) else ("", 0)

    def doc_of(pid):
        try:
            row = txt.loc[pid]
        except KeyError:
            return None
        title = (row["title"] or "").strip()
        abs = row["abstract"]
        abs_txt = (abs or "").strip()
        if not abs_txt:  # old papers: strip the "Title: ..." scaffold from text
            t = (row["text"] or "")
            abs_txt = t.split("Abstract:", 1)[-1].strip() if "Abstract:" in t else ""
        return {"pid": int(pid), "title": title, "abstract": abs_txt[:1200]}

    def retrieve(member_pids, k=TOPK):
        rows = [pid2row[int(p)] for p in member_pids if int(p) in pid2row]
        if len(rows) < 2:
            return []
        rows = np.array(rows)
        V = vn[rows]
        c = V.mean(0)
        c /= np.linalg.norm(c) + 1e-9
        sims = V @ c
        order = rows[np.argsort(-sims)[:k]]
        docs = [doc_of(int(pids[i])) for i in order]
        return [d for d in docs if d and (d["title"] or d["abstract"])]

    rng = np.random.default_rng(0)

    # root
    all_pids = pg.paper_id.values
    samp = rng.choice(all_pids, size=min(ROOT_SAMPLE, len(all_pids)), replace=False)
    tree = {"kind": "root", "ref": "whole corpus", "n": int(len(all_pids)),
            "docs": retrieve(samp), "children": []}

    for f in SPEC:
        fid = f["fid"]
        f_lbl, f_n = gt_of("main", fid)
        fnode = {"kind": "field", "ref": f_lbl, "n": f_n, "fid": fid,
                 "docs": retrieve(pg[pg.main_class_id == fid].paper_id.values),
                 "children": []}
        for d in f["divs"]:
            d_lbl, d_n = gt_of("division", d["c"])
            dnode = {"kind": "div", "ref": d["c"], "gt": d_lbl, "n": d_n,
                     "docs": retrieve(pg[pg.division == d["c"]].paper_id.values),
                     "children": []}
            for s in d["subs"]:
                s_lbl, s_n = gt_of("subdivision", s)
                snode = {"kind": "sub", "ref": s, "gt": s_lbl, "n": s_n,
                         "docs": retrieve(pg[pg.subdivision == s].paper_id.values)}
                dnode["children"].append(snode)
            fnode["children"].append(dnode)
        tree["children"].append(fnode)

    out = Path(__file__).resolve().parent / "nodes.json"
    out.write_text(json.dumps(tree, ensure_ascii=False))
    n = 1 + sum(1 + len(d["children"]) * 3 for d in [c for c in tree["children"]])
    print("wrote", out, "| nodes:",
          1 + len(tree["children"]) + sum(len(d["children"]) for d in tree["children"])
          + sum(len(s["children"]) for d in tree["children"] for s in d["children"]))
    for d in tree["children"]:
        print(f"  field {d['fid']:>2} {d['ref'][:30]:30} docs={len(d['docs'])}")


if __name__ == "__main__":
    main()
