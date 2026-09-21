# Idea compatibility: does doc2lora code-geometry track author-mobility FLOW beyond text?
#
# author mobility flow (PMI) over PACS codes  +  full-scale SBERT on all PACS papers
#   -> partial-Mantel of {gemma,mistral,qwen} / SBERT / TF-IDF code-similarity vs flow,
#      for all / cross-topic / within-topic code-pairs (+ permutation p, figure).

from os.path import join as j

COMPAT_DIR = j(APS_DIR, "compat")

compat_params = {
    "level": config.get("compat_level", ["fine"]),
    "topk":  config.get("compat_topk", [300]),
}
compat_ps = to_paramspace(compat_params)

FLOW_FILE   = j(COMPAT_DIR, f"flow_{compat_ps.wildcard_pattern}.npz")
COMPAT_JSON = j(COMPAT_DIR, f"results_{compat_ps.wildcard_pattern}.json")
COMPAT_CSV  = j(COMPAT_DIR, f"results_{compat_ps.wildcard_pattern}.csv")
COMPAT_FIG  = j(COMPAT_DIR, f"figure_{compat_ps.wildcard_pattern}.pdf")
SBERT_FULL  = j(EMB_DIR, "sbert_allmpnet.npz")

rule mobility_flow:
    output:
        flow=FLOW_FILE,
    params:
        author_net=config.get("aps_author_net", "/data/datasets/aps/preprocessed/paper_author_net.npz"),
        paper_table=config["aps_paper_table"],
        level=lambda w: w.level,
        topk=lambda w: w.topk,
        min_papers=config.get("compat_min_papers", 300),
    script:
        "../scripts/build_mobility_flow.py"

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

rule idea_compatibility_test:
    input:
        flow=FLOW_FILE,
        sbert=SBERT_FULL,
        paper_text=j(APS_DIR, "paper_text.parquet"),
        gemma=j(EMB_DIR, "gemma_norm_lora_emb.npz"),
        mistral=j(EMB_DIR, "mistral_norm_lora_emb.npz"),
        qwen=j(EMB_DIR, "qwen_norm_lora_emb.npz"),
    output:
        json=COMPAT_JSON,
        csv=COMPAT_CSV,
        fig=COMPAT_FIG,
    params:
        paper_table=config["aps_paper_table"],
        gene_paths=lambda w, input: {"gemma": input.gemma, "mistral": input.mistral, "qwen": input.qwen},
        n_perm=config.get("compat_n_perm", 10000),
        min_chars=config.get("compat_min_abstract_chars", 300),
        seed=config.get("compat_seed", 0),
    script:
        "../scripts/idea_compatibility.py"

rule idea_compatibility_all:
    input:
        expand(COMPAT_JSON, **compat_params),
        expand(COMPAT_FIG, **compat_params),
