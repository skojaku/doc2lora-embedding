"""Rebuild a citation sample's text pool from its triplet ids and the OpenAlex tables.

WHY THIS EXISTS. The transform the paper reports was trained on a citation sample of
42,332 tuples over 167,111 papers. `sample_edges.py` draws that sample AND writes its
text pool in one pass, so the pool looks like something only a re-draw can produce --
and a re-draw does not give the same sample back (see the GOTCHA in
workflow/rules/groupc_bench.smk: a later run overwrote the original with a different
draw, and only 29 of the 42,332 tuples survive in it).

But the pool is not independent of the sample. `sample_edges.py` writes exactly the
papers named in the triplets, formatted as "Title: <title>\\nAbstract: <abstract>".
So the small id file pins the large text file down completely: ship the ids, rebuild
the text. That is what makes the reported transform reproducible here rather than
merely archived.

The formatting below is copied from sample_edges.py and must stay identical to it; a
stray space changes every embedding downstream.

    python workflow/scripts/pool_text_from_triplets.py \\
        --triplets data/general_adapter/triplets_1x.parquet \\
        --out data/general_adapter/pool_text_1x.parquet
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench_data import openalex_prep_base  # noqa: E402

MIN_ABS = 200          # sample_edges.py's abstract-length floor, repeated here
CHUNK = 4_000_000


def main(triplets: str, out: str, prep: str | None = None):
    base = prep or openalex_prep_base()
    T = pd.read_parquet(triplets)
    wanted = set()
    for col in ("a", "pos", "neg_easy", "neg_hard"):
        wanted.update(int(x) for x in T[col].values)
    print(f"{len(T):,} triplets -> {len(wanted):,} distinct papers", flush=True)

    # abstracts: the same filtered scan sample_edges.py uses, so the same rows come back
    at = ds.dataset(f"{base}/abstracts.parquet").to_table(
        columns=["paper_id", "abstract"],
        filter=pc.field("paper_id").isin(sorted(wanted))).to_pandas()
    abstr = dict(zip(at.paper_id.values, at.abstract.values))
    print(f"  abstracts: {len(abstr):,}", flush=True)

    # titles: the paper table is too wide to hold, so stream it
    title = {}
    for ch in pd.read_csv(f"{base}/paper_table.csv", usecols=["paper_id", "title"],
                          chunksize=CHUNK):
        sub = ch[ch.paper_id.isin(wanted)]
        title.update(zip(sub.paper_id.values, sub.title.fillna("").values))
    print(f"  titles: {len(title):,}", flush=True)

    rows = []
    for pid in sorted(wanted):
        ab = str(abstr.get(pid, "") or "")
        if len(ab) < MIN_ABS:
            continue
        rows.append((int(pid), ("Title: " + str(title.get(pid, "")) + "\nAbstract: " + ab).strip()))
    df = pd.DataFrame(rows, columns=["pid", "text"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df):,} papers)", flush=True)
    if len(df) != len(wanted):
        print(f"NOTE: {len(wanted) - len(df):,} papers dropped by the {MIN_ABS}-char "
              f"abstract floor, as in the original draw", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--triplets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prep", default=None, help="OpenAlex preprocessed dir")
    a = ap.parse_args()
    main(a.triplets, a.out, a.prep)
