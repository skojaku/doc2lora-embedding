# Multi-field generalization of the idea-compatibility (author-mobility vs text) pipeline.
#
# Runs the SAME analysis as idea_compatibility.smk (APS/PACS) on arbitrary OpenAlex fields
# using the Scopus 2-level topic classification: sub_class = mobility "code" (252 subfields),
# main_class = coarse topic (26 fields) for the cross/within-topic pair split.
#
# Per field {field} in FIELD_LIST:
#   prepare_field_text -> paper_text.parquet + paper_topics.parquet
#   embed_field_papers (per model in MODEL_LIST: qwen/gemma/mistral genes)
#   field_sbert_abstracts (textual baseline)
#   field_mobility_flow (author sub_class->sub_class transition PMI)
#   field_idea_compatibility_test (partial-Mantel genes/SBERT/TF-IDF vs flow + figure)

from os.path import join as j

# ── parameters (from config; defaults if absent) ─────────────────────────
FIELD_LIST = config.get("FIELD_LIST", ["economics", "psychology", "chemistry"])
MODEL_LIST = config.get("MODELS", ["qwen", "gemma", "mistral"])

FIELDS_PREP_BASE = config.get(
    "fields_prep_base", "/data/projects/gravity-of-ideas/tmp-data/preprocessed")
FIELD_CAP = config.get("field_cap", 250000)
FIELD_SEED = config.get("field_seed", 0)
FIELD_MIN_TEXT = config.get("field_min_text_length", 300)

# host checkpoints (root config points at /opt docker paths; resolve host paths here)
FIELD_CKPT = {
    "qwen": config.get("qwen_host_checkpoint_path",
                       "data/agent_assets/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"),
    "gemma": config.get("gemma_host_checkpoint_path",
                        "data/agent_assets/gemma_demo/checkpoint-80000/pytorch_model.bin"),
    "mistral": config.get("mistral_host_checkpoint_path",
                          "data/agent_assets/mistral_7b_d2l/checkpoint-20000/pytorch_model.bin"),
}

FIELDS_DIR = j(DATA_DIR, "fields", "{field}")
FIELD_EMB_DIR = j(FIELDS_DIR, "embeddings")
FIELD_COMPAT_DIR = j(FIELDS_DIR, "compat")

field_compat_params = {
    "level": config.get("field_compat_level", ["sub"]),
    "topk": config.get("field_compat_topk", [150]),
}
field_compat_ps = to_paramspace(field_compat_params)

FIELD_PAPER_TEXT = j(FIELDS_DIR, "paper_text.parquet")
FIELD_PAPER_TOPICS = j(FIELDS_DIR, "paper_topics.parquet")
FIELD_SBERT = j(FIELD_EMB_DIR, "sbert_allmpnet.npz")
FIELD_GENE = j(FIELD_EMB_DIR, "{model}_norm_lora_emb.npz")
# scientific-text baselines folded into the compat figure (SBERT is already the primary text ctrl)
COMPAT_BASELINES = config.get("compat_baselines", ["specter2", "instructor", "text2vec"])
FIELD_BASELINE = j(FIELD_EMB_DIR, "baseline_{method}.npz")
FIELD_FLOW = j(FIELD_COMPAT_DIR, f"flow_{field_compat_ps.wildcard_pattern}.npz")
FIELD_COMPAT_JSON = j(FIELD_COMPAT_DIR, f"results_{field_compat_ps.wildcard_pattern}.json")
FIELD_COMPAT_CSV = j(FIELD_COMPAT_DIR, f"results_{field_compat_ps.wildcard_pattern}.csv")
FIELD_COMPAT_FIG = j(FIELD_COMPAT_DIR, f"figure_{field_compat_ps.wildcard_pattern}.pdf")

wildcard_constraints:
    field="|".join(FIELD_LIST),
    model="|".join(MODEL_LIST),


# ── data preparation ─────────────────────────────────────────────────────
rule prepare_field_text:
    output:
        paper_text=FIELD_PAPER_TEXT,
        paper_topics=FIELD_PAPER_TOPICS,
        report=j(FIELDS_DIR, "prepare_report.md"),
    params:
        field=lambda w: w.field,
        cap=FIELD_CAP,
        seed=FIELD_SEED,
        min_text_length=FIELD_MIN_TEXT,
    script:
        "../scripts/prepare_field_text.py"


# ── doc2lora gene embeddings (per model) ─────────────────────────────────
rule embed_field_papers:
    input:
        paper_text=FIELD_PAPER_TEXT,
    output:
        embeddings=FIELD_GENE,
    params:
        checkpoint_path=lambda w: FIELD_CKPT[w.model],
        shard_dir=j(FIELD_EMB_DIR, "{model}_shards"),
        shard_size=config.get("field_shard_size", config["shard_size"]),
        id_col="paper_id",
        max_length=config.get("field_max_length", 768),
        max_batch_tokens=lambda w: config.get("field_max_batch_tokens", {}).get(w.model, 16384),
        gpu_ids=config["gpu_ids"],
    resources:
        gpu=4,   # shards across all 4 GPUs internally -> reserve the whole machine
    priority: 100   # run embeddings (the long pole) back-to-back at full 4-GPU util;
                    # light gpu=1 jobs fill in at the end instead of fragmenting the budget
    script:
        "../scripts/embed_field_papers.py"


# ── SBERT textual baseline ───────────────────────────────────────────────
rule field_sbert_abstracts:
    input:
        paper_text=FIELD_PAPER_TEXT,
    output:
        embeddings=FIELD_SBERT,
    params:
        model_name=config.get("compat_sbert_model", "sentence-transformers/all-mpnet-base-v2"),
        min_chars=config.get("compat_min_abstract_chars", 300),
        shard_dir=j(FIELD_EMB_DIR, "sbert_shards"),
        shard_size=config.get("compat_sbert_shard_size", 20000),
    resources:
        gpu=1,
    script:
        "../scripts/embed_field_abstracts_sbert.py"


# ── author-mobility flow over sub_class codes ────────────────────────────
rule field_mobility_flow:
    input:
        paper_topics=FIELD_PAPER_TOPICS,
    output:
        flow=FIELD_FLOW,
    params:
        author_paper_table=lambda w: j(FIELDS_PREP_BASE, f"openalex-{w.field}", "author_paper_table.csv"),
        paper_table=lambda w: j(FIELDS_PREP_BASE, f"openalex-{w.field}", "paper_table.csv"),
        topk=lambda w: w.topk,
        min_papers=config.get("field_compat_min_papers", 1),
    script:
        "../scripts/build_field_mobility_flow.py"


# ── idea-compatibility test ──────────────────────────────────────────────
rule field_idea_compatibility_test:
    input:
        flow=FIELD_FLOW,
        sbert=FIELD_SBERT,
        paper_topics=FIELD_PAPER_TOPICS,
        paper_text=FIELD_PAPER_TEXT,
        genes=expand(FIELD_GENE, model=MODEL_LIST, allow_missing=True),
        baselines=expand(FIELD_BASELINE, method=COMPAT_BASELINES, allow_missing=True),
    output:
        json=FIELD_COMPAT_JSON,
        csv=FIELD_COMPAT_CSV,
        fig=FIELD_COMPAT_FIG,
    params:
        gene_paths=lambda w: {m: FIELD_GENE.format(field=w.field, model=m) for m in MODEL_LIST},
        baseline_paths=lambda w: {b: FIELD_BASELINE.format(field=w.field, method=b) for b in COMPAT_BASELINES},
        n_perm=config.get("compat_n_perm", 10000),
        min_chars=config.get("compat_min_abstract_chars", 300),
        seed=config.get("compat_seed", 0),
    script:
        "../scripts/idea_compatibility_field.py"


# ── aggregation ──────────────────────────────────────────────────────────
rule fields_all:
    input:
        expand(FIELD_COMPAT_JSON, field=FIELD_LIST, **field_compat_params),
        expand(FIELD_COMPAT_FIG, field=FIELD_LIST, **field_compat_params),
