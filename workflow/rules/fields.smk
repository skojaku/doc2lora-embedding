# The two OpenAlex field corpora the paper reports on, end to end.
#
# Economics and Psychology stand next to APS physics so that no claim rests on one
# corpus. Scopus two-level topics give the label for the topic-classification task
# (sub_class, 252 subfields) and the coarse split (main_class, 26 fields).
#
# Per field {field} in FIELD_LIST:
#   prepare_field_text     -> paper_text.parquet + paper_topics.parquet
#   embed_field_papers     -> one gene matrix per model in MODEL_LIST
#   field_sbert_abstracts  -> the SBERT vectors the score pools read

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
FIELD_PAPER_TEXT = j(FIELDS_DIR, "paper_text.parquet")
FIELD_PAPER_TOPICS = j(FIELDS_DIR, "paper_topics.parquet")
FIELD_SBERT = j(FIELD_EMB_DIR, "sbert_allmpnet.npz")
FIELD_GENE = j(FIELD_EMB_DIR, "{model}_norm_lora_emb.npz")

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


# ── aggregation ──────────────────────────────────────────────────────────
rule fields_all:
    input:
        expand(FIELD_SBERT, field=FIELD_LIST),
        expand(FIELD_GENE, field=FIELD_LIST, model=MODEL_LIST),
