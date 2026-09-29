"""Join APS paper titles with OpenAlex abstracts via DOI matching.

Usage (standalone):
    python prepare_aps_text.py \
        --aps-papers /data/datasets/aps/preprocessed/paper_table.csv \
        --openalex-papers /data/datasets/openalex/preprocessed/paper_table.csv \
        --openalex-abstracts /data/datasets/openalex/preprocessed/abstracts.parquet \
        --output data/aps/paper_text.parquet \
        --report data/aps/matching_report.md

Snakemake: reads input/output/params from snakemake object.
"""

import argparse
import pandas as pd
from pathlib import Path


def main(aps_path, openalex_papers_path, openalex_abstracts_path, output_path, report_path):
    # Load APS papers
    aps = pd.read_csv(aps_path, usecols=["paper_id", "doi", "title", "journal_code"])
    aps = aps.rename(columns={"paper_id": "aps_paper_id"})
    n_total = len(aps)

    # Normalize APS DOIs to match OpenAlex format: lowercase, prepend https://doi.org/
    aps["doi_normalized"] = "https://doi.org/" + aps["doi"].str.lower()

    # Load OpenAlex paper table (only need paper_id and doi)
    oa_papers = pd.read_csv(openalex_papers_path, usecols=["paper_id", "doi"], dtype={"doi": str})
    oa_papers = oa_papers.rename(columns={"paper_id": "openalex_paper_id"})
    oa_papers["doi"] = oa_papers["doi"].str.lower()
    # Drop duplicate DOIs in OpenAlex (keep first match)
    oa_papers = oa_papers.dropna(subset=["doi"]).drop_duplicates(subset=["doi"], keep="first")

    # Join APS → OpenAlex on normalized DOI
    merged = aps.merge(oa_papers, left_on="doi_normalized", right_on="doi", how="left", suffixes=("", "_oa"))
    n_doi_matched = merged["openalex_paper_id"].notna().sum()

    # Load OpenAlex abstracts
    abstracts = pd.read_parquet(openalex_abstracts_path)
    abstracts = abstracts.rename(columns={"paper_id": "openalex_paper_id"})

    # Join with abstracts
    merged = merged.merge(abstracts, on="openalex_paper_id", how="left")
    n_abstract_matched = merged["abstract"].notna().sum()

    # Build text column
    has_abstract = merged["abstract"].notna()
    merged["text"] = "Title: " + merged["title"].fillna("")
    merged.loc[has_abstract, "text"] = (
        "Title: " + merged.loc[has_abstract, "title"].fillna("")
        + "\nAbstract: " + merged.loc[has_abstract, "abstract"]
    )

    # Select output columns and sort by aps_paper_id
    result = merged[["aps_paper_id", "doi", "title", "abstract", "text", "journal_code"]].copy()
    result = result.sort_values("aps_paper_id").reset_index(drop=True)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)

    # Generate report
    n_no_title = result["title"].isna().sum()
    report = f"""# APS–OpenAlex Matching Report

| Metric | Count | % |
|---|---:|---:|
| Total APS papers | {n_total:,} | 100.0% |
| DOI matched to OpenAlex | {n_doi_matched:,} | {100*n_doi_matched/n_total:.1f}% |
| Abstract found | {n_abstract_matched:,} | {100*n_abstract_matched/n_total:.1f}% |
| Title-only (no abstract) | {n_total - n_abstract_matched:,} | {100*(n_total - n_abstract_matched)/n_total:.1f}% |
| Missing title | {n_no_title:,} | {100*n_no_title/n_total:.1f}% |
"""
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)

    print(report)
    print(f"Saved {len(result):,} rows to {output_path}")


if __name__ == "__main__":
    try:
        # Snakemake mode
        main(
            aps_path=snakemake.input.aps_papers,
            openalex_papers_path=snakemake.input.openalex_papers,
            openalex_abstracts_path=snakemake.input.openalex_abstracts,
            output_path=snakemake.output.paper_text,
            report_path=snakemake.output.report,
        )
    except NameError:
        parser = argparse.ArgumentParser()
        parser.add_argument("--aps-papers", required=True)
        parser.add_argument("--openalex-papers", required=True)
        parser.add_argument("--openalex-abstracts", required=True)
        parser.add_argument("--output", required=True)
        parser.add_argument("--report", required=True)
        args = parser.parse_args()
        main(args.aps_papers, args.openalex_papers, args.openalex_abstracts, args.output, args.report)
