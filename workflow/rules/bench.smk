# BENCH: sliced eval-subset of every embedding file (NO recompute), so the benchmark workflow runs off
# small files instead of the full 0.5-1.8M-paper corpora. make_subset.py slices the existing npz files
# (doc2lora genes, genkron, kron, sbert/specter2/instructor, icae) down to the eval-touched paper union
# (collect_field_ids.py) and writes them to data/bench/<field>/embeddings. The full source datasets --
# including the standalone full APS at data/aps used by another experiment -- are left untouched.
#
# The benchmark eval reads the subset via BENCH_ROOT (fields) / BENCH_S2_PROC (S2AND).
# RUN: snakemake bench_all --rerun-triggers mtime -j2
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

rule bench_slice_s2and:
    output: marker=j("data", "bench", "s2and", "{ds}", ".sliced"),
    resources: mem_gb=8,
    shell: f"python {SCRIPTS}/make_subset.py s2and {{wildcards.ds}} && touch {{output.marker}}"

# 3) evaluate on the subset (suffix _bench). Same harness, BENCH_ROOT/BENCH_S2_PROC point at the slices.
rule bench_eval_field:
    input: marker=j("data", "bench", "{field}", "embeddings", ".sliced"),
           topic=j(IC, "{field}_topic_ids.parquet"), fp=j(IC, "{field}_fut_pool.parquet"),
           cohorts=j(IC, "{field}_cohorts.parquet"),
    output: csv=j(KR, "results_{field}_{enc}_bench.csv"),
    resources: gpu=1, mem_gb=40,
    shell: f"{IC_ENV} BENCH_ROOT=data/bench INCLUDE_ICAE=1 OUT_SUFFIX=_bench "
           f"TOPIC_IDS_FILE={IC}/{{wildcards.field}}_topic_ids.parquet "
           f"NP_FUT_POOL_FILE={IC}/{{wildcards.field}}_fut_pool.parquet "
           f"NP_COHORT_FILE={IC}/{{wildcards.field}}_cohorts.parquet "
           f"{GPU} python {SCRIPTS}/eval_all.py --field {{wildcards.field}} --enc {{wildcards.enc}}"

rule bench_eval_s2and:
    input: marker=j("data", "bench", "s2and", "{ds}", ".sliced"),
    output: csv=j(S2, "results_{ds}_{enc}_bench.csv"),
    resources: gpu=1, mem_gb=20,
    shell: f"{IC_ENV} BENCH_S2_PROC=data/bench/s2and OUT_SUFFIX=_bench KRON_STEPS=2000 KRON_REG=10 "
           f"{GPU} python {SCRIPTS}/and_eval.py {{wildcards.ds}} {{wildcards.enc}}"


def _bench_targets():
    out = []
    for f, encs in BENCH_FIELD_ENC.items():
        out += [j(KR, f"results_{f}_{e}_bench.csv") for e in encs]
    out += [j(S2, f"results_{ds}_qwen_bench.csv") for ds in BENCH_S2AND]
    return out

rule bench_all:
    input: _bench_targets()
