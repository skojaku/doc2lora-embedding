# BENCH: the sliced eval subset the temporal-hardening chain runs on (App. datasets).
#
# collect_field_ids.py computes the union of papers any benchmark actually touches, and
# make_subset.py slices the existing embedding files down to it -- no recompute -- into
# data/bench/<field>/embeddings. groupc_bench reads that through BENCH_ROOT, which is
# what keeps the decontamination test off the full 0.5-1.8M-paper corpora.
#
# RUN: snakemake bench_subsets -j2
import os
from os.path import join as j

IC = "data/icae"
KR = "data/kron"
S2 = "data/s2and"
BENCH_FIELDS = ["economics", "psychology", "aps", "arxiv_cs", "arxiv_math"]
BENCH_S2AND = ["zbmath", "qian", "arnetminer", "pubmed", "kisti"]
# encoders to report per field (only those with gene files present are evaluated; eval_all skips missing)
BENCH_FIELD_ENC = {"economics": ["gemma", "qwen", "mistral"], "psychology": ["gemma", "qwen", "mistral"],
                   "aps": ["gemma", "qwen", "mistral"], "arxiv_cs": ["gemma", "qwen"],
                   "arxiv_math": ["gemma", "qwen"]}
IC_ENV = (f"set -a; source .env 2>/dev/null; set +a; "
          f"export HF_HOME={os.path.abspath('data/agent_assets/hf_cache')} "
          f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;")
GPU = f"bash {SCRIPTS}/gpu_run.sh"

wildcard_constraints:
    field="|".join(BENCH_FIELDS),
    ds="|".join(BENCH_S2AND),
    enc="gemma|qwen|mistral",


# 1) eval-id subset (topic+np+collab union, fixed topic + negative pool) — CPU
rule bench_ids:
    output:
        ids=j(IC, "{field}_eval_ids.parquet"),
        topic=j(IC, "{field}_topic_ids.parquet"),
        fp=j(IC, "{field}_fut_pool.parquet"),
        cohorts=j(IC, "{field}_cohorts.parquet"),
    resources: mem_gb=80,
    shell: f"python {SCRIPTS}/collect_field_ids.py {{wildcards.field}}"

# 2) slice every embedding npz -> data/bench/<field>/embeddings (marker output; file set varies)
rule bench_slice_field:
    input: ids=j(IC, "{field}_eval_ids.parquet"),
    output: marker=j("data", "bench", "{field}", "embeddings", ".sliced"),
    resources: mem_gb=40,
    shell: f"python {SCRIPTS}/make_subset.py field {{wildcards.field}} && touch {{output.marker}}"

rule bench_subsets:
    input:
        expand(j("data", "bench", "{field}", "embeddings", ".sliced"), field=BENCH_FIELDS),
