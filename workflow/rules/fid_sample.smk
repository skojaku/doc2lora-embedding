# The paper-abstract sample that the prompt-sensitivity chain decodes (App. psens).
#
# One rule, because one rule is all the manuscript reads from here: a fixed,
# seeded draw of APS abstracts long enough to be worth decoding. `psens_decode`
# takes this sample so that every paraphrase is scored on the same documents.
from os.path import join as j

FID_DIR = j(DATA_DIR, "groupc", "fidelity")
FID_SAMPLE = j(FID_DIR, "sample.parquet")


rule fid_sample:
    input:
        paper_text=ancient(j(APS_DIR, "paper_text.parquet")),
    output:
        sample=FID_SAMPLE,
    params:
        n_papers=config.get("fid_n_papers", 200),
        seed=config.get("fid_seed", 0),
        min_chars=config.get("fid_min_abstract_chars", 400),
    resources:
        mem_gb=16,
    script:
        "../scripts/groupc/fid_sample.py"
