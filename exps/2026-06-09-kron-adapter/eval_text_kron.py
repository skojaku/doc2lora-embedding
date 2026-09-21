"""Compare RAW text vs text+citation-kron (and recall gene+kron) on the 3 tasks, same harness.
Both raw and kron text use author_cos / score_auc / knn on their OWN vecs (apples-to-apples, no parquet).

Usage: CUDA_VISIBLE_DEVICES=1 python exps/2026-06-09-kron-adapter/eval_text_kron.py --field economics
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, "workflow/scripts"); sys.path.insert(0, "exps/2026-06-09-kron-adapter")
from bench_data import load, aps_paper_table
from eval_all import collab_eval, np_eval, knn_topic

TEXT = {"sbert": "sbert_allmpnet.npz", "specter2": "baseline_specter2.npz", "instructor": "baseline_instructor.npz"}


def main(a):
    pt, edges, emb_dir = load(a.field)
    yr = dict(zip(pt.paper_id.astype(int), pt.year)); edges = edges.copy()
    edges["year"] = edges.paper_id.map(yr); edges = edges.dropna(subset=["year"]); edges["year"] = edges.year.astype(int)
    dcol = pd.read_parquet(f"data/collab_scores_{a.field}_hard.parquet").reset_index(drop=True)

    # build comparison list: each text encoder raw + its _kron
    encs = []
    for t, fn in TEXT.items():
        encs.append((t, fn, "vecs"))
        kf = f"{t}_kron.npz"
        if os.path.exists(f"{emb_dir}/{kf}"):
            encs.append((f"{t}_kron", kf, "vecs"))

    # topic subsample: labeled papers common to all involved files
    if a.field == "aps":
        jt = pd.read_csv(aps_paper_table(), usecols=["paper_id", "journal_code"]).dropna()
        codes = {c: i for i, c in enumerate(sorted(jt.journal_code.unique()))}
        lab = {int(p): codes[c] for p, c in zip(jt.paper_id, jt.journal_code)}
    else:
        tp = pd.read_parquet(f"data/fields/{a.field}/paper_topics.parquet")
        lab = dict(zip(tp.paper_id.astype(int), tp.main_class.astype(int)))
    common = None
    for _, fn, _ in encs:
        zp = set(int(p) for p in np.load(f"{emb_dir}/{fn}", allow_pickle=True)["paper_ids"])
        common = zp if common is None else (common & zp)
    cand = np.array([p for p in common if p in lab]); rng = np.random.default_rng(0); rng.shuffle(cand)
    cand = cand[:a.topic_n]; ntr = int(0.8 * len(cand)); tr_pids, te_pids = cand[:ntr], cand[ntr:]

    rows = []
    for name, fn, key in encs:
        cau, cap = collab_eval(a.field, emb_dir, edges, name, fn, key, True, dcol)   # is_gene=True -> author_cos on vecs
        npauc = np_eval(a.field, emb_dir, edges, fn, key)
        tf1, tacc = knn_topic(a.field, emb_dir, fn, key, lab, tr_pids, te_pids)
        rows.append(dict(model=name, collab_AUC=cau, np_AUC=npauc, topic_F1=tf1))
        print(f"  {name:14} collab={cau:.3f} | np={npauc:.3f} | topic_F1={tf1:.3f}", flush=True)

    print("\n===== TEXT raw vs +citation-kron (" + a.field + ") =====")
    print(pd.DataFrame(rows).to_string(index=False))
    pd.DataFrame(rows).to_csv(f"exps/2026-06-09-kron-adapter/results_text_kron_{a.field}.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--field", default="economics"); p.add_argument("--topic_n", type=int, default=60000)
    main(p.parse_args())
