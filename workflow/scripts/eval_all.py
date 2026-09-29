"""Evaluate {gemma, gemma_kron, sbert, specter2, instructor} on three tasks for one field:
  (1) collab  : future co-authorship link prediction (hard negatives), AUC/AP over windows
  (2) np      : next-paper ranking, pooled AUC
  (3) topic   : main_class classification, cosine-kNN macro-F1 / accuracy (GPU)
Reuses the verified harness in workflow/scripts + data/layer_bands.

Usage: CUDA_VISIBLE_DEVICES=1 python workflow/scripts/eval_all.py --field economics
"""
import argparse, os, sys, json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

sys.path.insert(0, "workflow/scripts")
sys.path.insert(0, "data/layer_bands")
from bench_data import load, aps_paper_table
from eval_collab2 import author_cos
from nextpaper_layerbands import build_authors, score_auc

def encoder_list(enc):  # name, file, key, is_gene
    return [
        (enc,          f"{enc}_norm_lora_emb.npz", "embeddings", True),
        (f"{enc}_kron", f"{enc}_kron_emb.npz",     "embeddings", True),
        ("sbert",      "sbert_allmpnet.npz",      "vecs",       False),
        ("specter2",   "baseline_specter2.npz",   "vecs",       False),
        ("instructor", "baseline_instructor.npz", "vecs",       False),
    ]
WINMAP = {"aps": [2000, 2004, 2008], "economics": [2008, 2012, 2016], "psychology": [2008, 2012, 2016],
          "arxiv_math": [2010, 2014, 2018], "arxiv_cs": [2010, 2014, 2018]}
DY = 3


# ---------- collab ----------
def collab_eval(field, emb_dir, edges, name, fn, key, is_gene, d):
    """gene encoders: author_cos over matrix; text encoders: precomputed parquet cosine column."""
    au, ap = [], []
    if not is_gene:
        if name not in d.columns:
            return np.nan, np.nan
        for _, g in d.dropna(subset=[name]).groupby("window"):
            if g.y.nunique() > 1:
                au.append(roc_auc_score(g.y, g[name])); ap.append(average_precision_score(g.y, g[name]))
        return float(np.mean(au)), float(np.mean(ap))
    z = np.load(f"{emb_dir}/{fn}", allow_pickle=True); E = z[key].astype(np.float32)
    if E.ndim > 2: E = E.reshape(E.shape[0], -1)
    prow = {int(p): i for i, p in enumerate(z["paper_ids"])}
    windows = sorted(d.window.unique())
    perwin = {t: edges[edges.year.between(t - DY, t)].groupby("author_id")["paper_id"]
              .apply(lambda s: tuple(int(x) for x in s)).to_dict() for t in windows}
    for t in windows:
        sub = d[d.window == t]
        cand = list(zip(sub.a.astype(int), sub.b.astype(int)))
        s, cov = author_cos(cand, perwin[t], E, prow)
        m = cov & ~np.isnan(s)
        if m.sum() > 10 and sub.y.values[m].std() > 0:
            au.append(roc_auc_score(sub.y.values[m], s[m])); ap.append(average_precision_score(sub.y.values[m], s[m]))
    del E, z
    return (float(np.mean(au)) if au else np.nan), (float(np.mean(ap)) if ap else np.nan)


# ---------- next-paper ----------
def np_eval(field, emb_dir, edges, fn, key):
    z = np.load(f"{emb_dir}/{fn}", allow_pickle=True); E = z[key].astype(np.float32)
    if E.ndim > 2: E = E.reshape(E.shape[0], -1)
    prow = {int(p): i for i, p in enumerate(z["paper_ids"])}
    coh, fut = build_authors(field, edges, prow)
    v = float(score_auc(E, prow, coh, fut)); del E, z
    return v


# ---------- topic (cosine kNN on GPU) ----------
def knn_topic(field, emb_dir, fn, key, labels_by_pid, tr_pids, te_pids, k=10):
    z = np.load(f"{emb_dir}/{fn}", allow_pickle=True)
    prow = {int(p): i for i, p in enumerate(z["paper_ids"].astype(np.int64))}
    idx_tr = np.array([prow[p] for p in tr_pids]); idx_te = np.array([prow[p] for p in te_pids])
    Etr = torch.tensor(np.asarray(z[key][idx_tr]).reshape(len(idx_tr), -1), dtype=torch.float16, device="cuda")
    Ete = torch.tensor(np.asarray(z[key][idx_te]).reshape(len(idx_te), -1), dtype=torch.float16, device="cuda")
    Etr = torch.nn.functional.normalize(Etr.float(), dim=1).half()
    Ete = torch.nn.functional.normalize(Ete.float(), dim=1).half()
    ytr = np.array([labels_by_pid[int(p)] for p in tr_pids])
    yte = np.array([labels_by_pid[int(p)] for p in te_pids])
    preds = np.empty(len(idx_te), np.int64)
    ytr_t = torch.tensor(ytr, device="cuda")
    for i in range(0, len(idx_te), 2048):
        sims = Ete[i:i + 2048] @ Etr.t()                # [b, Ntr]
        nn = sims.topk(k, dim=1).indices                # [b, k]
        vote = ytr_t[nn]                                # [b, k]
        b = vote.shape[0]
        # majority vote
        preds[i:i + b] = torch.mode(vote, dim=1).values.cpu().numpy()
    del z
    return float(f1_score(yte, preds, average="macro")), float(accuracy_score(yte, preds))


def main(a):
    pt, edges, emb_dir = load(a.field)
    broot = os.environ.get("BENCH_ROOT")     # read sliced subset embeddings from data/bench/<field>/embeddings
    if broot:
        emb_dir = f"{broot}/{a.field}/embeddings"
    yr = dict(zip(pt.paper_id.astype(int), pt.year)); edges = edges.copy()
    edges["year"] = edges.paper_id.map(yr); edges = edges.dropna(subset=["year"]); edges["year"] = edges.year.astype(int)
    dcol = pd.read_parquet(f"data/collab_scores_{a.field}_hard.parquet").reset_index(drop=True)

    ENCODERS = encoder_list(a.enc)
    gk = f"{a.enc}_genkron_emb.npz"      # GENERAL (cross-field) adapter applied to this field's genes
    if os.path.exists(f"{emb_dir}/{gk}"):
        ENCODERS.insert(2, (f"{a.enc}_genkron", gk, "embeddings", True))
    # ICAE baseline + invertible per-token ICAE adapter (encoder-agnostic; subset coverage).
    # is_gene=True so collab/np/topic all use the matrix path (ICAE has no precomputed collab column).
    if os.environ.get("INCLUDE_ICAE"):
        for nm, fn in [("icae", "icae_emb.npz"), ("icae_genkron", "icae_genkron_emb.npz")]:
            if os.path.exists(f"{emb_dir}/{fn}"):
                ENCODERS.append((nm, fn, "vecs", True))
    # gte-large = Text-to-LoRA's NATIVE coordinates (#151). Its hypernetwork reads a frozen
    # gte vector and expands it into an adapter, so this row is that encoder's own space and
    # must be labelled as such, never as an adapter space. Not a 2024+-encoder addition on
    # its merits -- decision D6 of #142 closed #69 the other way.
    # is_gene=True for the same reason as ICAE: the matrix path, since there is no
    # precomputed collab cosine column for it.
    if os.environ.get("INCLUDE_GTE"):
        fn = "baseline_gte_large.npz"
        if os.path.exists(f"{emb_dir}/{fn}"):
            ENCODERS.append(("gte_large", fn, "vecs", True))
    # topic labels + a fixed subsample (gene-covered papers) shared across encoders by ROW index
    if a.field == "aps":
        jt = pd.read_csv(aps_paper_table(), usecols=["paper_id", "journal_code"]).dropna()
        codes = {c: i for i, c in enumerate(sorted(jt.journal_code.unique()))}
        lab = {int(p): codes[c] for p, c in zip(jt.paper_id, jt.journal_code)}
    else:
        tp = pd.read_parquet(f"data/fields/{a.field}/paper_topics.parquet")
        lab = dict(zip(tp.paper_id.astype(int), tp.main_class.astype(int)))
    # common paper_ids present in EVERY encoder (coverage differs across encoders on APS) AND labeled
    common = None
    for _, fn, _, _ in ENCODERS:
        if not os.path.exists(f"{emb_dir}/{fn}"):
            continue
        zp = set(int(p) for p in np.load(f"{emb_dir}/{fn}", allow_pickle=True)["paper_ids"])
        common = zp if common is None else (common & zp)
    tif = os.environ.get("TOPIC_IDS_FILE")
    if tif:   # FIXED topic subsample (so ICAE need only cover these papers; identical set for every method)
        fixed = pd.read_parquet(tif)["paper_id"].astype(int).values
        cand = np.array([int(p) for p in fixed if p in common and p in lab])
    else:
        cand = np.array([p for p in common if p in lab])
        rng = np.random.default_rng(0); rng.shuffle(cand)
        cand = cand[:a.topic_n]
    ntr = int(0.8 * len(cand)); tr_pids, te_pids = cand[:ntr], cand[ntr:]
    print(f"[topic] {len(cand)} labeled papers common to all encoders ({ntr} train / {len(cand)-ntr} test)", flush=True)

    rows = []
    for name, fn, key, is_gene in ENCODERS:
        if not os.path.exists(f"{emb_dir}/{fn}"):
            print(f"[skip] {name} (missing {fn})", flush=True); continue
        cau, cap = collab_eval(a.field, emb_dir, edges, name, fn, key, is_gene, dcol)
        npauc = np_eval(a.field, emb_dir, edges, fn, key)
        tf1, tacc = knn_topic(a.field, emb_dir, fn, key, lab, tr_pids, te_pids)
        rows.append(dict(model=name, collab_AUC=cau, collab_AP=cap, np_AUC=npauc, topic_F1=tf1, topic_acc=tacc))
        print(f"  {name:12} collab AUC={cau:.3f} AP={cap:.3f} | np AUC={npauc:.3f} | "
              f"topic F1={tf1:.3f} acc={tacc:.3f}", flush=True)

    df = pd.DataFrame(rows)
    print("\n===== SUMMARY (" + a.field + ") =====")
    print(df.to_string(index=False))
    suf = os.environ.get("OUT_SUFFIX", "")
    df.to_csv(f"data/kron/results_{a.field}_{a.enc}{suf}.csv", index=False)
    print(f"[saved] data/kron/results_{a.field}_{a.enc}{suf}.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--field", default="economics")
    p.add_argument("--enc", default="gemma")
    p.add_argument("--topic_n", type=int, default=60000)
    main(p.parse_args())
