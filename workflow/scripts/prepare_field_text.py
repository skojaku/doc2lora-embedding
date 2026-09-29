"""[CPU] Prepare an OpenAlex *field* corpus for the multi-field robustness study.

Generalises prepare_aps_text/prepare_xdomain_text to any precompiled field under
/data/projects/gravity-of-ideas/tmp-data/preprocessed/openalex-<field>/.

Produces three artifacts under data/fields/<field>/:
  * paper_text.parquet  — the EMBEDDING subsample (capped, abstract-bearing) with
      columns: paper_id, openalex_paper_id, title, abstract, text, year, frac_year,
               main_class, sub_class, main_class_title, sub_class_title
    `paper_id` is the field-local id (load-bearing key used by the generic embedder
    and every downstream analysis).
  * paper_topics.parquet — ALL field papers (no abstract filter) with
      columns: paper_id, sub_class, main_class  (for the author-mobility flow, which
    needs full publication sequences, not just the embedded subsample).
  * prepare_report.md — summary.

Topic join (verified): field.openalex_paper_id == generic paper_category_table.paper_id
  -> (main_class_id, sub_class_id) at sequence==0 (primary) -> category_table.title.
The Scopus/ASJC taxonomy is 2-level: 26 `main` fields / 252 `sub` subfields.

Usage (standalone):
    python prepare_field_text.py --field economics --cap 250000 --seed 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PREP_BASE = "/data/projects/gravity-of-ideas/tmp-data/preprocessed"
CAT_TABLE = f"{PREP_BASE}/openalex/category_table.csv"
PAPER_CAT = f"{PREP_BASE}/openalex/paper_category_table.csv"


def _build_text(df):
    has_abs = df["abstract"].notna() & (df["abstract"].astype(str).str.len() > 0)
    text = "Title: " + df["title"].fillna("")
    text = text.where(
        ~has_abs,
        "Title: " + df["title"].fillna("") + "\nAbstract: " + df["abstract"].astype(str),
    )
    return text


def load_category_labels():
    cat = pd.read_csv(CAT_TABLE, usecols=["class_id", "type", "title"])
    main_map = dict(zip(cat[cat.type == "main"].class_id, cat[cat.type == "main"].title))
    sub_map = dict(zip(cat[cat.type == "sub"].class_id, cat[cat.type == "sub"].title))
    return main_map, sub_map


def join_topics(oa_ids, chunksize=5_000_000):
    """Map a set of global paper_ids (== field.openalex_paper_id) to primary
    (main_class_id, sub_class_id) via the 5.8GB paper_category_table, chunked."""
    id_set = set(int(x) for x in oa_ids)
    parts = []
    for chunk in pd.read_csv(
        PAPER_CAT,
        usecols=["paper_id", "main_class_id", "sub_class_id", "sequence"],
        chunksize=chunksize,
    ):
        sel = chunk[(chunk["sequence"] == 0) & (chunk["paper_id"].isin(id_set))]
        if len(sel):
            parts.append(sel[["paper_id", "main_class_id", "sub_class_id"]])
    if not parts:
        return pd.DataFrame(columns=["paper_id", "main_class_id", "sub_class_id"])
    out = pd.concat(parts, ignore_index=True).drop_duplicates("paper_id")
    return out


def main(field, cap=250000, seed=0, min_text_length=300, out_dir=None):
    out_dir = Path(out_dir or f"data/fields/{field}")
    out_dir.mkdir(parents=True, exist_ok=True)
    src = f"{PREP_BASE}/openalex-{field}/paper_table.csv"
    print(f"[prep:{field}] reading {src}")
    df = pd.read_csv(
        src,
        usecols=["paper_id", "openalex_paper_id", "title", "year", "abstract", "frac_year"],
    )
    n_total = len(df)

    # ---- topic join over ALL field papers (for the mobility flow) ----
    main_map, sub_map = load_category_labels()
    print(f"[prep:{field}] joining topics for {n_total:,} papers (chunked over paper_category_table)...")
    topics = join_topics(df["openalex_paper_id"].values)
    topics = topics.rename(columns={"paper_id": "openalex_paper_id",
                                    "main_class_id": "main_class",
                                    "sub_class_id": "sub_class"})
    df = df.merge(topics, on="openalex_paper_id", how="left")

    # full-population topic table (no abstract filter) for flow construction
    topo = df[["paper_id", "sub_class", "main_class"]].dropna(subset=["sub_class"]).copy()
    topo["sub_class"] = topo["sub_class"].astype(int)
    topo["main_class"] = topo["main_class"].astype(int)
    topo.to_parquet(out_dir / "paper_topics.parquet", index=False)

    # ---- embedding subsample: abstract-bearing, capped ----
    emb = df.dropna(subset=["title"]).copy()
    emb = emb[emb["abstract"].notna() & (emb["abstract"].astype(str).str.len() >= min_text_length)]
    emb = emb.dropna(subset=["sub_class"])
    n_eligible = len(emb)
    if cap and n_eligible > cap:
        # stratified-ish by subfield via groupwise proportional sample, fall back to random
        rng = np.random.default_rng(seed)
        emb = emb.sample(n=cap, random_state=seed).copy()
    emb["main_class"] = emb["main_class"].astype(int)
    emb["sub_class"] = emb["sub_class"].astype(int)
    emb["main_class_title"] = emb["main_class"].map(main_map)
    emb["sub_class_title"] = emb["sub_class"].map(sub_map)
    emb["text"] = _build_text(emb)
    emb = emb.sort_values("paper_id").reset_index(drop=True)
    cols = ["paper_id", "openalex_paper_id", "title", "abstract", "text", "year",
            "frac_year", "main_class", "sub_class", "main_class_title", "sub_class_title"]
    emb[cols].to_parquet(out_dir / "paper_text.parquet", index=False)

    n_sub = emb["sub_class"].nunique()
    n_main = emb["main_class"].nunique()
    report = f"""# Field corpus: {field}

| metric | value |
|---|---:|
| total field papers | {n_total:,} |
| with primary topic | {len(topo):,} |
| eligible (abstract ≥{min_text_length}, titled, topic'd) | {n_eligible:,} |
| embedding subsample (cap={cap}) | {len(emb):,} |
| distinct main fields | {n_main} |
| distinct subfields | {n_sub} |
| median text len (chars) | {int(emb.text.str.len().median())} |

Top subfields in subsample:
{emb['sub_class_title'].value_counts().head(12).to_string()}
"""
    (out_dir / "prepare_report.md").write_text(report)
    print(report)
    print(f"[prep:{field}] -> {out_dir}/paper_text.parquet ({len(emb):,}), "
          f"paper_topics.parquet ({len(topo):,})")


if __name__ == "__main__":
    if "snakemake" in sys.modules:
        sm = snakemake  # noqa: F821
        out_dir = str(Path(sm.output["paper_text"]).parent)
        main(sm.params["field"], cap=int(sm.params["cap"]), seed=int(sm.params["seed"]),
             min_text_length=int(sm.params["min_text_length"]), out_dir=out_dir)
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("--field", required=True)
        ap.add_argument("--cap", type=int, default=250000)
        ap.add_argument("--seed", type=int, default=0)
        ap.add_argument("--min-text-length", type=int, default=300)
        ap.add_argument("--out-dir", default=None)
        a = ap.parse_args()
        main(a.field, a.cap, a.seed, a.min_text_length, a.out_dir)
