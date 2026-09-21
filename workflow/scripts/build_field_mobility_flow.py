"""Author-mobility flow between Scopus sub_class codes = external 'idea compatibility' signal,
generalised from build_mobility_flow.py (PACS) to any OpenAlex field.

For each author, order papers by frac_year and count consecutive sub_class->sub_class
transitions (code_t -> code_{t+1}); symmetrize + PMI-normalize (excess mobility vs chance,
removing code popularity). Career flow ACROSS papers, not within-paper co-occurrence.

Restrict to the top-K sub_class codes by paper count (many fields have far fewer than the
252 Scopus subfields active). Coarse split for cross/within-topic uses main_class (26 fields).

Dual-mode (Snakemake + CLI). Out: npz {codes (sub_class str), code_main (main_class per code),
counts, flow_pmi, code_npapers, level="sub"}.
"""
import argparse
import sys

import numpy as np
import pandas as pd


def main(paper_topics, author_paper_table, paper_table, out_flow, topk=150, min_papers=1):
    topo = pd.read_parquet(paper_topics, columns=["paper_id", "sub_class", "main_class"])
    topo["sub_class"] = topo["sub_class"].astype(int)
    topo["main_class"] = topo["main_class"].astype(int)

    # frac_year per paper from the field paper_table
    pt = pd.read_csv(paper_table, usecols=["paper_id", "frac_year"])
    topo = topo.merge(pt, on="paper_id", how="left")

    # top-K sub_class codes by paper count (handle fields with < topk active subfields)
    vc = topo["sub_class"].value_counts()
    vc = vc[vc >= min_papers]
    top = vc.head(topk).index.tolist()
    code_id = {int(c): i for i, c in enumerate(top)}
    npap = vc.reindex(top).to_numpy()
    K = len(top)

    # main_class per code (modal main_class among that sub_class's papers)
    sub2main = (
        topo[topo["sub_class"].isin(code_id)]
        .groupby("sub_class")["main_class"]
        .agg(lambda s: int(s.mode().iloc[0]))
    )
    code_main = np.array([int(sub2main.loc[c]) for c in top], dtype=np.int64)

    # paper -> (code idx, frac_year) lookup
    sub = topo[topo["sub_class"].isin(code_id)].copy()
    sub["cidx"] = sub["sub_class"].map(code_id).astype(int)
    paper_code = dict(zip(sub["paper_id"].to_numpy(), sub["cidx"].to_numpy()))
    paper_year = dict(zip(sub["paper_id"].to_numpy(), sub["frac_year"].to_numpy()))

    ap = pd.read_csv(author_paper_table, usecols=["paper_id", "author_id"])
    ap = ap[ap["paper_id"].isin(paper_code)]

    T = np.zeros((K, K), np.float64)
    for _, grp in ap.groupby("author_id", sort=False):
        papers = grp["paper_id"].to_numpy()
        if len(papers) < 2:
            continue
        ys = np.array([paper_year.get(p, np.nan) for p in papers])
        cs = np.array([paper_code[p] for p in papers])
        ok = ~np.isnan(ys)
        if ok.sum() < 2:
            continue
        cseq = cs[ok][np.argsort(ys[ok], kind="stable")]
        for i in range(len(cseq) - 1):
            T[cseq[i], cseq[i + 1]] += 1.0

    Ts = T + T.T
    tot = Ts.sum()
    row = Ts.sum(1, keepdims=True)
    expected = (row @ row.T) / (tot + 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((Ts + 1e-9) / (expected + 1e-9))
    np.fill_diagonal(pmi, 0.0)

    np.savez(out_flow,
             codes=np.array([str(c) for c in top], dtype=object),
             code_main=code_main, counts=T, flow_pmi=pmi,
             code_npapers=npap, level="sub")
    print(f"[flow] level=sub K={K} sub_class codes (>= {min_papers} papers), "
          f"{int(T.sum())} transitions, {int(T.sum()-np.trace(T))} cross-code -> {out_flow}")


if __name__ == "__main__":
    if "snakemake" in sys.modules:
        sm = snakemake  # noqa: F821
        main(sm.input["paper_topics"], sm.params["author_paper_table"],
             sm.params["paper_table"], sm.output["flow"],
             topk=int(sm.params["topk"]), min_papers=int(sm.params["min_papers"]))
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("--paper-topics", required=True)
        ap.add_argument("--author-paper-table", required=True)
        ap.add_argument("--paper-table", required=True)
        ap.add_argument("--out-flow", required=True)
        ap.add_argument("--topk", type=int, default=150)
        ap.add_argument("--min-papers", type=int, default=1)
        a = ap.parse_args()
        main(a.paper_topics, a.author_paper_table, a.paper_table, a.out_flow,
             a.topk, a.min_papers)
