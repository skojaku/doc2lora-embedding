# Full-corpus SBERT vectors over the APS abstracts.
#
# `all-mpnet-base-v2` is both a text baseline in its own right and the scorer the
# decode metrics use, so this is the one rule the rest of the workflow reads from
# here: `unc_pool_field` scores it alongside the genes.
from os.path import join as j

SBERT_FULL = j(EMB_DIR, "sbert_allmpnet.npz")


rule sbert_abstracts:
    input:
        paper_text=j(APS_DIR, "paper_text.parquet"),
    output:
        embeddings=SBERT_FULL,
    params:
        paper_table=config["aps_paper_table"],
        model_name=config.get("compat_sbert_model", "sentence-transformers/all-mpnet-base-v2"),
        min_chars=config.get("compat_min_abstract_chars", 300),
        shard_dir=j(EMB_DIR, "sbert_shards"),
        shard_size=config.get("compat_sbert_shard_size", 20000),
    resources:
        gpu=1,
    script:
        "../scripts/embed_abstracts_sbert.py"
