# ICAE baseline + invertible per-token ICAE adapter, evaluated on the SAME benchmarks as doc2lora.
# ICAE is encoder-agnostic (one Mistral-7B model): compress a doc -> 128 memory slots (4096-dim),
# mean-pool -> the raw `icae` embedding. The adapter (kron.KronAdapter L=128 d=4096: common rotation+
# scale shared across tokens + per-token scaling) is trained on the SAME OpenAlex citation triplets as
# doc2lora's general adapter, applied per memory token, then pooled -> `icae_genkron`. Invertible, so
# the compressed memory stays decodable (decode_roundtrip.py).
#
# Generating the whole field corpora (565k-987k papers) is unrealistic; collect_field_ids.py computes
# the MINIMAL paper-id union each benchmark touches (~250-440k/field) and we embed only that.
#
# RUN: snakemake icae_all --rerun-triggers mtime -j2
import os
from os.path import join as j

IC = "data/icae"
KR = "data/kron"
S2 = "data/s2and"
IC_FIELDS = ["economics", "psychology"]          # OpenAlex field tasks (eval_all)
IC_S2AND = ["zbmath", "qian", "arnetminer", "pubmed", "kisti"]
IC_ENV = (f"set -a; source .env 2>/dev/null; set +a; "
          f"export HF_HOME={os.path.abspath('data/agent_assets/hf_cache')} "
          f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;")
GPU = f"bash {SCRIPTS}/gpu_run.sh"
OAX_EMB = j("data", "fields", "{field}", "embeddings")
APS_EMB = j("data", "aps", "embeddings")

wildcard_constraints:
    field="economics|psychology",
    ds="|".join(IC_S2AND),


# ---- minimal touched-id subsets (CPU) ----
rule ic_collect_ids:
    output: expand(j(IC, "{f}_eval_ids.parquet"), f=IC_FIELDS + ["aps"]),
            expand(j(IC, "{f}_topic_ids.parquet"), f=IC_FIELDS + ["aps"]),
    resources: mem_gb=80,
    shell: f"python {SCRIPTS}/collect_field_ids.py {' '.join(IC_FIELDS)} aps"

# ---- ICAE-compress the OpenAlex pool -> per-token slots (for adapter training) ----
rule ic_pool_slots:
    input: j("data/general_adapter", "pool_text.parquet"),
    output: slots=j(IC, "pool_slots.npy"), pids=j(IC, "pool_pids.npy"),
    resources: gpu=1, mem_gb=40,
    shell: f"{IC_ENV} BATCH=24 {GPU} python {SCRIPTS}/embed_pool_slots.py"

rule ic_train:
    input: slots=j(IC, "pool_slots.npy"), trip=j("data/general_adapter", "triplets.parquet"),
    output: adapter=j(IC, "adapter_icae.pt"),
    resources: gpu=1, mem_gb=40,
    shell: f"{IC_ENV} STEPS=8000 {GPU} python {SCRIPTS}/train_icae_adapter.py"

# ---- embed each benchmark subset (raw icae + icae_genkron), needs the adapter ----
rule ic_embed_field:
    input: adapter=j(IC, "adapter_icae.pt"), ids=j(IC, "{field}_eval_ids.parquet"),
    output: raw=j(OAX_EMB, "icae_emb.npz"), gk=j(OAX_EMB, "icae_genkron_emb.npz"),
    resources: gpu=1, mem_gb=40,
    shell: f"{IC_ENV} BATCH=24 {GPU} python {SCRIPTS}/embed_apply.py field {{wildcards.field}}"

rule ic_embed_aps:
    input: adapter=j(IC, "adapter_icae.pt"), ids=j(IC, "aps_eval_ids.parquet"),
    output: raw=j(APS_EMB, "icae_emb.npz"), gk=j(APS_EMB, "icae_genkron_emb.npz"),
    resources: gpu=1, mem_gb=40,
    shell: f"{IC_ENV} BATCH=24 {GPU} python {SCRIPTS}/embed_apply.py aps"

rule ic_embed_s2and:
    input: adapter=j(IC, "adapter_icae.pt"),
    output: raw=j(S2, "proc", "{ds}", "icae.npz"), gk=j(S2, "proc", "{ds}", "icae_genkron.npz"),
    resources: gpu=1, mem_gb=20,
    shell: f"{IC_ENV} BATCH=24 {GPU} python {SCRIPTS}/embed_apply.py s2and {{wildcards.ds}}"

# ---- evaluate (INCLUDE_ICAE adds icae/icae_genkron to the comparison; fixed topic subsample) ----
rule ic_eval_field:
    input: raw=j(OAX_EMB, "icae_emb.npz"), gk=j(OAX_EMB, "icae_genkron_emb.npz"),
           topic=j(IC, "{field}_topic_ids.parquet"),
    output: csv=j(KR, "results_{field}_qwen_icae.csv"),
    resources: gpu=1, mem_gb=60,
    shell: f"{IC_ENV} INCLUDE_ICAE=1 OUT_SUFFIX=_icae TOPIC_IDS_FILE={IC}/{{wildcards.field}}_topic_ids.parquet "
           f"NP_FUT_POOL_FILE={IC}/{{wildcards.field}}_fut_pool.parquet "
           f"NP_COHORT_FILE={IC}/{{wildcards.field}}_cohorts.parquet "
           f"{GPU} python {SCRIPTS}/eval_all.py --field {{wildcards.field}} --enc qwen"

rule ic_eval_aps:
    input: raw=j(APS_EMB, "icae_emb.npz"), gk=j(APS_EMB, "icae_genkron_emb.npz"),
           topic=j(IC, "aps_topic_ids.parquet"),
    output: csv=j(KR, "results_aps_qwen_icae.csv"),
    resources: gpu=1, mem_gb=60,
    shell: f"{IC_ENV} INCLUDE_ICAE=1 OUT_SUFFIX=_icae TOPIC_IDS_FILE={IC}/aps_topic_ids.parquet "
           f"NP_FUT_POOL_FILE={IC}/aps_fut_pool.parquet "
           f"NP_COHORT_FILE={IC}/aps_cohorts.parquet "
           f"{GPU} python {SCRIPTS}/eval_all.py --field aps --enc qwen"

rule ic_eval_s2and:
    input: raw=j(S2, "proc", "{ds}", "icae.npz"), gk=j(S2, "proc", "{ds}", "icae_genkron.npz"),
    output: csv=j(S2, "results_{ds}_qwen_icae.csv"),
    resources: gpu=1, mem_gb=40,
    shell: f"{IC_ENV} OUT_SUFFIX=_icae KRON_STEPS=2000 KRON_REG=10 {GPU} python {SCRIPTS}/and_eval.py {{wildcards.ds}} qwen"


rule icae_all:
    input:
        expand(j(KR, "results_{field}_qwen_icae.csv"), field=IC_FIELDS),
        j(KR, "results_aps_qwen_icae.csv"),
        expand(j(S2, "results_{ds}_qwen_icae.csv"), ds=IC_S2AND),
