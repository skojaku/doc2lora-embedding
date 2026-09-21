"""Author-mobility flow between PACS codes = external 'idea compatibility' signal.

For each author, order papers by year and count consecutive PACS1-code transitions
(code_t -> code_{t+1}); symmetrize + PMI-normalize (excess mobility vs chance, removing
code popularity). This is career flow ACROSS papers, not within-paper co-occurrence.

Dual-mode (Snakemake + CLI). Out: flow npz {codes, counts, flow_pmi, code_npapers, level}.
"""
import argparse
import re
import sys
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from bench_data import aps_author_net, aps_paper_table  # noqa: E402

import numpy as np
import pandas as pd
import scipy.sparse as sp

PACS_RE = re.compile(r"^\d\d\.\d\d")


def main(author_net, paper_table, out_flow, level="fine", topk=300, min_papers=300):
    width = 5 if level == "fine" else 2
    d = np.load(author_net, allow_pickle=True)
    PA = sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
    AP = PA.T.tocsr()                                    # authors x papers
    nrows = PA.shape[0]

    pt = pd.read_csv(paper_table, usecols=["paper_id", "frac_year", "PACS1"],
                     dtype={"paper_id": np.int64, "PACS1": str})
    pt = pt[pt["PACS1"].notna() & pt["PACS1"].str.match(PACS_RE, na=False)].copy()
    pt["code"] = pt["PACS1"].str[:width]
    vc = pt["code"].value_counts()
    top = vc[vc >= min_papers].head(topk).index.tolist()
    code_id = {c: i for i, c in enumerate(top)}
    npap = vc.reindex(top).to_numpy()
    K = len(top)

    code_arr = np.full(nrows, -1, np.int32)
    year_arr = np.full(nrows, np.nan)
    sub = pt[pt["paper_id"] < nrows].copy()
    sub["cidx"] = sub["code"].map(code_id)
    pids = sub["paper_id"].to_numpy()
    year_arr[pids] = sub["frac_year"].to_numpy()
    m = sub["cidx"].notna().to_numpy()
    code_arr[pids[m]] = sub["cidx"][m].astype(int).to_numpy()

    T = np.zeros((K, K), np.float64)
    for a in range(AP.shape[0]):
        papers = AP.indices[AP.indptr[a]:AP.indptr[a + 1]]
        if len(papers) < 2:
            continue
        ys, cs = year_arr[papers], code_arr[papers]
        ok = (~np.isnan(ys)) & (cs >= 0)
        if ok.sum() < 2:
            continue
        cseq = cs[ok][np.argsort(ys[ok], kind="stable")]
        for i in range(len(cseq) - 1):
            T[cseq[i], cseq[i + 1]] += 1.0

    Ts = T + T.T
    tot = Ts.sum()
    row = Ts.sum(1, keepdims=True)
    expected = (row @ row.T) / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((Ts + 1e-9) / (expected + 1e-9))
    np.fill_diagonal(pmi, 0.0)

    np.savez(out_flow, codes=np.array(top, dtype=object), counts=T, flow_pmi=pmi,
             code_npapers=npap, level=level)
    print(f"[flow] level={level} K={K} codes (>= {min_papers} papers), "
          f"{int(T.sum())} transitions, {int(T.sum()-np.trace(T))} cross-code -> {out_flow}")


if __name__ == "__main__":
    if "snakemake" in sys.modules:
        sm = snakemake  # noqa: F821
        main(sm.params["author_net"], sm.params["paper_table"], sm.output["flow"],
             level=sm.params["level"], topk=int(sm.params["topk"]),
             min_papers=int(sm.params["min_papers"]))
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("--author-net", default=aps_author_net())
        ap.add_argument("--paper-table", default=aps_paper_table())
        ap.add_argument("--out-flow", required=True)
        ap.add_argument("--level", default="fine", choices=["fine", "major"])
        ap.add_argument("--topk", type=int, default=300)
        ap.add_argument("--min-papers", type=int, default=300)
        a = ap.parse_args()
        main(a.author_net, a.paper_table, a.out_flow, a.level, a.topk, a.min_papers)
