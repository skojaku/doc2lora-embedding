# GENERAL doc2lora kron adapter: trained on ~600k sampled OpenAlex citation pairs (cross-field), then
# applied to every benchmark and evaluated (next-paper/topic/collab field tasks + S2AND name disambiguation).
# Reuses existing field/S2AND genes (which ARE doc2lora embeddings) — gemma/qwen pool economics+psychology+
# chemistry; mistral uses chemistry (only field with mistral genes). Eval re-uses eval_all.py / and_eval.py
# with OUT_SUFFIX=_general and the genkron embedding auto-included.
#
# RUN: snakemake general --rerun-triggers mtime -j2   (after the s2and run frees the GPUs)
import os
from os.path import join as j

GA = "data/general_adapter"
S2 = "data/s2and"
KR = "data/kron"
GA_ENC = config.get("ga_encoders", ["gemma", "qwen", "mistral"])
GA_OAX_FIELD = {"economics": ["gemma", "qwen", "mistral"], "psychology": ["gemma", "qwen", "mistral"]}  # openalex field tasks
GA_APS_ENC = ["gemma", "qwen", "mistral"]
GA_S2AND = config.get("ga_s2and", ["zbmath", "qian", "arnetminer", "pubmed", "kisti"])

GA_ENV = (f"set -a; source .env 2>/dev/null; set +a; "
          f"export HF_HOME={os.path.abspath('data/agent_assets/hf_cache')} "
          f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;")
GPU = f"bash {SCRIPTS}/gpu_run.sh"
OAX_EMB2 = j("data", "fields", "{field}", "embeddings")
APS_EMB2 = j("data", "aps", "embeddings")

wildcard_constraints:
    enc="gemma|qwen|mistral",
    field="economics|psychology",
    ds="|".join(GA_S2AND),


# fresh broad sample from the raw OpenAlex edge lists -> citation triplets (hard+easy negs) + pool text
rule ga_sample_edges:
    output: trip=j(GA, "triplets.parquet"), pool=j(GA, "pool_text.parquet"),
    resources: mem_gb=120,
    shell: f"python {SCRIPTS}/sample_edges.py"

# fresh doc2lora embedding of the pool, per encoder
rule ga_embed_pool:
    input: pool=j(GA, "pool_text.parquet"),
    output: npz=j(GA, "pool_genes_{enc}.npz"),
    params: batch=lambda w: "16" if w.enc == "mistral" else "32",
    resources: gpu=1, mem_gb=40,
    shell: f"{GA_ENV} MAX_TOKENS=512 BATCH={{params.batch}} {GPU} python {SCRIPTS}/embed_pool.py {{wildcards.enc}}"

rule ga_train:
    input: trip=j(GA, "triplets.parquet"), pool=j(GA, "pool_genes_{enc}.npz"),
    output: adapter=j(GA, "adapter_general_{enc}.pt"),
    resources: gpu=1, mem_gb=40,
    shell: f"{GA_ENV} {GPU} python {SCRIPTS}/train_general.py {{wildcards.enc}}"


# ---- apply general adapter to each benchmark's genes ----
rule ga_apply_oax:
    input: adapter=j(GA, "adapter_general_{enc}.pt"), gene=j(OAX_EMB2, "{enc}_norm_lora_emb.npz"),
    output: gk=j(OAX_EMB2, "{enc}_genkron_emb.npz"),
    resources: gpu=1, mem_gb=60,
    shell: f"{GA_ENV} {GPU} python {SCRIPTS}/apply_general.py {{wildcards.enc}} {{input.gene}} {{output.gk}} embeddings"

rule ga_apply_aps:
    input: adapter=j(GA, "adapter_general_{enc}.pt"), gene=j(APS_EMB2, "{enc}_norm_lora_emb.npz"),
    output: gk=j(APS_EMB2, "{enc}_genkron_emb.npz"),
    resources: gpu=1, mem_gb=60,
    wildcard_constraints: enc="|".join(GA_APS_ENC),
    shell: f"{GA_ENV} {GPU} python {SCRIPTS}/apply_general.py {{wildcards.enc}} {{input.gene}} {{output.gk}} embeddings"

rule ga_apply_s2and:
    input: adapter=j(GA, "adapter_general_{enc}.pt"), gene=j(S2, "proc", "{ds}", "genes_{enc}.npz"),
    output: gk=j(S2, "proc", "{ds}", "genes_{enc}_genkron.npz"),
    resources: gpu=1, mem_gb=30,
    shell: f"{GA_ENV} {GPU} python {SCRIPTS}/apply_general.py {{wildcards.enc}} {{input.gene}} {{output.gk}} embeddings"


# ---- evaluate (general genkron auto-included; _general suffix) ----
rule ga_eval_oax:
    input:
        gk=j(OAX_EMB2, "{enc}_genkron_emb.npz"), gene=j(OAX_EMB2, "{enc}_norm_lora_emb.npz"),
        sbert=j(OAX_EMB2, "sbert_allmpnet.npz"), specter2=j(OAX_EMB2, "baseline_specter2.npz"),
        instructor=j(OAX_EMB2, "baseline_instructor.npz"),
        collab=j("data", "collab_scores_{field}_hard.parquet"),
        topics=j("data", "fields", "{field}", "paper_topics.parquet"),
    output: csv=j(KR, "results_{field}_{enc}_general.csv"),
    resources: gpu=1, mem_gb=60,
    shell: f"{GA_ENV} OUT_SUFFIX=_general {GPU} python {SCRIPTS}/eval_all.py --field {{wildcards.field}} --enc {{wildcards.enc}}"

rule ga_eval_aps:
    input:
        gk=j(APS_EMB2, "{enc}_genkron_emb.npz"), gene=j(APS_EMB2, "{enc}_norm_lora_emb.npz"),
        sbert=j(APS_EMB2, "sbert_allmpnet.npz"), specter2=j(APS_EMB2, "baseline_specter2.npz"),
        instructor=j(APS_EMB2, "baseline_instructor.npz"),
        collab=j("data", "collab_scores_aps_hard.parquet"),
    output: csv=j(KR, "results_aps_{enc}_general.csv"),
    resources: gpu=1, mem_gb=60,
    wildcard_constraints: enc="|".join(GA_APS_ENC),
    shell: f"{GA_ENV} OUT_SUFFIX=_general {GPU} python {SCRIPTS}/eval_all.py --field aps --enc {{wildcards.enc}}"

rule ga_eval_s2and:
    input:
        gk=j(S2, "proc", "{ds}", "genes_{enc}_genkron.npz"), gene=j(S2, "proc", "{ds}", "genes_{enc}.npz"),
        spec=j(S2, "proc", "{ds}", "specter.npz"), sbert=j(S2, "proc", "{ds}", "sbert.npz"),
        instr=j(S2, "proc", "{ds}", "instructor.npz"),
    output: csv=j(S2, "results_{ds}_{enc}_general.csv"),
    resources: gpu=1, mem_gb=40,
    shell: f"{GA_ENV} OUT_SUFFIX=_general KRON_STEPS=2000 KRON_REG=10 {GPU} python {SCRIPTS}/and_eval.py {{wildcards.ds}} {{wildcards.enc}}"


# ---- leakage audit (issue #22): overlap between the g_theta training pool and
# every eval split. CPU-only, post-hoc; reads saved IDs, no retraining. The
# OpenAlex master paper_table (for the APS DOI map) is an external source file,
# passed as a param rather than tracked. ----
rule ga_leakage_overlap:
    input:
        triplets=j(GA, "triplets.parquet"),
        pool=j(GA, "pool_text.parquet"),
        eco=j("data", "fields", "economics", "paper_text.parquet"),
        psy=j("data", "fields", "psychology", "paper_text.parquet"),
        aps=j("data", "aps", "paper_text.parquet"),
        s2and=expand(j(S2, "proc", "{ds}", "paper_text.parquet"),
                     ds=["zbmath", "qian", "arnetminer", "pubmed", "kisti"]),
    params:
        oa_master=config.get("openalex_master_paper_table",
                             "/data/datasets/openalex/preprocessed/paper_table.csv"),
    output:
        json=j(GA, "leakage_overlap.json"),
    resources:
        mem_gb=16,
    script:
        "../../workflow/scripts/compute_leakage_overlap.py"


rule ga_leakage:
    input:
        rules.ga_leakage_overlap.output.json,


def _ga_targets():
    out = []
    for f, encs in GA_OAX_FIELD.items():
        for e in encs:
            out.append(j(KR, f"results_{f}_{e}_general.csv"))
    for e in GA_APS_ENC:
        out.append(j(KR, f"results_aps_{e}_general.csv"))
    for ds in GA_S2AND:
        for e in GA_ENC:
            out.append(j(S2, f"results_{ds}_{e}_general.csv"))
    return out

rule ga_all:
    input: _ga_targets(),
