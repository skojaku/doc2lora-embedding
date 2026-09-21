"""[CPU] Idea-compatibility test for an OpenAlex field, generalised from idea_compatibility.py
(APS/PACS) to the Scopus 2-level taxonomy.

Per-sub_class centroids from ALL field papers for every encoder (doc2lora genes + SBERT + TF-IDF
+ optional extra baselines: SPECTER2/INSTRUCTOR/text2vec). Mantel + PARTIAL Mantel of code-
similarity vs author-mobility FLOW, controlling text, with permutation p-values, for ALL /
CROSS-TOPIC / WITHIN-TOPIC code-pairs (cross = different main_class). Plus a figure.

Keying: gene/baseline npz are keyed by field-local paper_id; paper_topics paper_id is the same
field-local id -> join directly (no DOI crosswalk).

Dual-mode. Out: results json + tidy csv + figure pdf.
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def simmat(C):
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    return C @ C.T


def resid(y, x):
    xr, yr = rankdata(x), rankdata(y)
    b = np.polyfit(xr, yr, 1)
    return yr - (b[0] * xr + b[1])


def partial(a, b, ctrl):
    return float(spearmanr(resid(a, ctrl), resid(b, ctrl)).correlation)


def _npz_centroids(path, by_code, cidx, K, emb_key=None, id_key="paper_ids"):
    """Centroids per code from a npz keyed by field-local paper_id. Auto-detects the
    embedding key: genes use 'embeddings', SBERT/text baselines use 'vecs'."""
    d = np.load(path, allow_pickle=True)
    if emb_key is None:
        emb_key = "embeddings" if "embeddings" in d.files else "vecs"
    emb, pids = d[emb_key], d[id_key]
    if emb.ndim > 2:
        emb = emb.reshape(emb.shape[0], -1)
    row = {int(p): i for i, p in enumerate(pids)}
    C = np.zeros((K, emb.shape[1]), np.float32)
    for c, plist in by_code.items():
        r = [row[int(p)] for p in plist if int(p) in row]
        if r:
            C[cidx[c]] = unit(emb[r].astype(np.float32).mean(0))
    del emb
    return C


def main(flow_npz, sbert_npz, gene_paths, paper_topics, paper_text, out_json, out_csv, out_fig,
         baseline_paths=None, n_perm=10000, min_chars=300, seed=0):
    baseline_paths = baseline_paths or {}
    rng = np.random.default_rng(seed)
    fl = np.load(flow_npz, allow_pickle=True)
    codes = [str(c) for c in fl["codes"]]
    flow = fl["flow_pmi"]
    level = str(fl["level"]) if "level" in fl else "sub"
    code_main = {codes[i]: int(m) for i, m in enumerate(fl["code_main"])}
    cidx = {c: i for i, c in enumerate(codes)}
    K = len(codes)
    major = np.array([code_main[c] for c in codes])

    # code -> field-local paper_ids (ALL field papers)
    topo = pd.read_parquet(paper_topics, columns=["paper_id", "sub_class"])
    topo["sub_class"] = topo["sub_class"].astype(int).astype(str)
    topo = topo[topo["sub_class"].isin(cidx)]
    by_code = topo.groupby("sub_class")["paper_id"].apply(list)

    # ---- SBERT centroids ({paper_ids, vecs})
    sz = np.load(sbert_npz, allow_pickle=True)
    srow = {int(p): i for i, p in enumerate(sz["paper_ids"])}
    SV = sz["vecs"].astype(np.float32)
    Cs = np.zeros((K, SV.shape[1]), np.float32)
    for c, plist in by_code.items():
        r = [srow[int(p)] for p in plist if int(p) in srow]
        if r:
            Cs[cidx[c]] = unit(SV[r].mean(0))
    del SV

    # ---- TF-IDF centroids (fit on all valid abstracts in the embedding subsample)
    tx = pd.read_parquet(paper_text, columns=["paper_id", "abstract"])
    tx = tx[tx["abstract"].str.len().fillna(0) >= min_chars]
    valid_pids = set(int(p) for plist in by_code for p in plist)
    tx = tx[tx["paper_id"].isin(valid_pids)].drop_duplicates("paper_id").reset_index(drop=True)
    trow = {int(p): i for i, p in enumerate(tx["paper_id"])}
    from sklearn.feature_extraction.text import TfidfVectorizer
    TF = TfidfVectorizer(stop_words="english", max_features=40000).fit_transform(tx["abstract"])
    Ct = np.zeros((K, TF.shape[1]), np.float32)
    for c, plist in by_code.items():
        r = [trow[int(p)] for p in plist if int(p) in trow]
        if r:
            Ct[cidx[c]] = unit(np.asarray(TF[r].mean(0)).ravel())

    Ds, Dt = simmat(Cs), simmat(Ct)
    iu = np.triu_indices(K, 1)
    cross = major[iu[0]] != major[iu[1]]
    # skip empty/tiny masks (single-discipline corpora e.g. arXiv-math have no cross-topic pairs)
    masks = {k: v for k, v in {"all": np.ones(len(iu[0]), bool),
                               "cross_topic": cross, "within_topic": ~cross}.items()
             if int(v.sum()) >= 10}
    f = flow[iu]; s = Ds[iu]; t = Dt[iu]

    def perm_p(qm, sm, mask, observed):
        if n_perm <= 0:
            return None
        c = 0
        for _ in range(n_perm):
            p = rng.permutation(K)
            fp = flow[np.ix_(p, p)][iu][mask]
            if abs(partial(qm, fp, sm)) >= abs(observed):
                c += 1
        return (c + 1) / (n_perm + 1)

    res = {"level": level, "n_codes": K,
           "n_pairs": {m: int(mask.sum()) for m, mask in masks.items()},
           "text_vs_flow": {}, "encoders": {}}
    for m, mask in masks.items():
        res["text_vs_flow"][m] = {"sbert_flow": float(spearmanr(s[mask], f[mask]).correlation),
                                  "tfidf_flow": float(spearmanr(t[mask], f[mask]).correlation)}

    Dq_cache = {}
    # genes + extra baselines, all in {embeddings, paper_ids} format keyed by field-local paper_id
    enc_paths = {**{k: v for k, v in gene_paths.items()},
                 **{k: v for k, v in baseline_paths.items()}}
    for name, path in enc_paths.items():
        C = _npz_centroids(path, by_code, cidx, K)
        Dq = simmat(C); Dq_cache[name] = Dq; q = Dq[iu]
        res["encoders"][name] = {}
        for m, mask in masks.items():
            fm, qm, sm, tm = f[mask], q[mask], s[mask], t[mask]
            obs = partial(qm, fm, sm)
            res["encoders"][name][m] = {
                "d2l_flow_raw": float(spearmanr(qm, fm).correlation),
                "d2l_given_sbert": obs, "sbert_given_d2l": partial(sm, fm, qm),
                "d2l_given_tfidf": partial(qm, fm, tm), "tfidf_given_d2l": partial(tm, fm, qm),
                "mantel_p_d2l_given_sbert": perm_p(qm, sm, mask, obs)}
        print(f"[compat] {name} done")

    # examples: high-flow / low-SBERT code pairs for the strongest encoder
    em = "cross_topic" if "cross_topic" in masks else "all"   # single-discipline corpora have no cross_topic
    if res["encoders"]:
        best = max(res["encoders"], key=lambda n: res["encoders"][n][em]["d2l_given_sbert"])
        Dq = Dq_cache[best]
        fr, sr = rankdata(f), rankdata(s)
        emask = cross if em == "cross_topic" else np.ones(len(f), bool)
        score = np.where(emask, fr / len(f) - sr / len(s), -9)
        ex = []
        for k in np.argsort(-score)[:30]:
            i, j = iu[0][k], iu[1][k]
            ex.append({"a": codes[i], "b": codes[j], "flow_pmi": round(float(flow[i, j]), 2),
                       "sbert": round(float(Ds[i, j]), 3), "d2l": round(float(Dq[i, j]), 3),
                       "tfidf": round(float(Dt[i, j]), 3), "encoder": best})
        res["examples_highflow_lowtext_cross"] = ex

    json.dump(res, open(out_json, "w"), indent=2)
    rows = []
    for name in res["encoders"]:
        for m in masks:
            for k, v in res["encoders"][name][m].items():
                rows.append({"encoder": name, "mask": m, "metric": k, "value": v})
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    enc = list(res["encoders"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, m in zip(axes, masks):
        x = np.arange(len(enc)); w = 0.35
        ax.bar(x - w / 2, [res["encoders"][e][m]["d2l_given_sbert"] for e in enc], w, label="enc│sbert")
        ax.bar(x + w / 2, [res["encoders"][e][m]["sbert_given_d2l"] for e in enc], w, label="sbert│enc")
        ax.axhline(0, c="k", lw=.5); ax.set_xticks(x); ax.set_xticklabels(enc, rotation=45, ha="right")
        ax.set_title(m)
    axes[0].set_ylabel("partial Spearman with mobility flow")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Idea compatibility: code-geometry vs author-mobility flow beyond text "
                 f"({level}, {K} codes)")
    fig.tight_layout()
    fig.savefig(out_fig)
    print(f"[compat] -> {out_json} / {out_csv} / {out_fig}")
    print(f"\n{'encoder/mask':<22}{'raw':>7}{'|sbert':>8}{'sbert|d':>9}{'|tfidf':>8}{'tfidf|d':>9}{'p':>8}")
    for name in enc:
        for m in masks:
            r = res["encoders"][name][m]
            p = r["mantel_p_d2l_given_sbert"]
            print(f"{name+'/'+m:<22}{r['d2l_flow_raw']:>7.3f}{r['d2l_given_sbert']:>8.3f}"
                  f"{r['sbert_given_d2l']:>9.3f}{r['d2l_given_tfidf']:>8.3f}{r['tfidf_given_d2l']:>9.3f}"
                  f"{(p if p is not None else float('nan')):>8.4f}")


if __name__ == "__main__":
    if "snakemake" in sys.modules:
        sm = snakemake  # noqa: F821
        bl = dict(sm.params["baseline_paths"]) if sm.params.get("baseline_paths") else {}
        main(sm.input["flow"], sm.input["sbert"], dict(sm.params["gene_paths"]),
             sm.input["paper_topics"], sm.input["paper_text"],
             sm.output["json"], sm.output["csv"], sm.output["fig"],
             baseline_paths=bl, n_perm=int(sm.params["n_perm"]),
             min_chars=int(sm.params["min_chars"]), seed=int(sm.params["seed"]))
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("--flow", required=True)
        ap.add_argument("--sbert", required=True)
        ap.add_argument("--paper-topics", required=True)
        ap.add_argument("--paper-text", required=True)
        ap.add_argument("--gemma")
        ap.add_argument("--mistral")
        ap.add_argument("--qwen")
        ap.add_argument("--baseline", action="append", default=[],
                        help="name=path.npz extra baseline ({embeddings,paper_ids}); repeatable")
        ap.add_argument("--out-json", required=True)
        ap.add_argument("--out-csv", required=True)
        ap.add_argument("--out-fig", required=True)
        ap.add_argument("--n-perm", type=int, default=10000)
        ap.add_argument("--min-chars", type=int, default=300)
        ap.add_argument("--seed", type=int, default=0)
        a = ap.parse_args()
        genes = {k: v for k, v in
                 {"gemma": a.gemma, "mistral": a.mistral, "qwen": a.qwen}.items() if v}
        bl = {}
        for spec in a.baseline:
            name, path = spec.split("=", 1)
            bl[name] = path
        main(a.flow, a.sbert, genes, a.paper_topics, a.paper_text,
             a.out_json, a.out_csv, a.out_fig, baseline_paths=bl,
             n_perm=a.n_perm, min_chars=a.min_chars, seed=a.seed)
