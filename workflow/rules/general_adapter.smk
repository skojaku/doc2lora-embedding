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


# THE SAMPLE THE PAPER REPORTS IS THE 1x ONE. A later data-scaling run re-drew the
# citation sample and overwrote triplets.parquet with a larger, DIFFERENT draw: it is
# not a superset, and only 29 of the reported 42,332 tuples survive in it. So the
# reported transform -- and the ICAE transform that has to match it -- read the 1x
# files, and this workflow treats those as the canonical ones.
#
# The 1x triplet ids ship with the artifact bundle (1.3 MB). Its text pool does not,
# because the pool is not independent evidence: sample_edges.py writes exactly the
# papers the triplets name, so the ids pin the text down and pool_text_1x is rebuilt
# from them. That is what makes the reported transform reproducible rather than
# merely archived.
GA_SAMPLE = config.get("ga_sample", "_1x")            # "" re-draws instead
# Obtained, not derived: this is the one file the reported transform cannot be
# rebuilt without, so it is an input to the workflow like the corpora are. It ships
# in the `results` artifact tier (1.3 MB, ids only).
GA_TRIPLETS = config.get("ga_triplets", j(GA, f"triplets{GA_SAMPLE}.parquet"))
GA_POOL_TEXT = j(GA, f"pool_text{GA_SAMPLE}.parquet")


# A fresh broad draw from the raw OpenAlex edge lists. Seeded (default_rng(0)) but NOT
# the reported sample: it writes the unsuffixed names, and re-running it is how the
# reported one was lost.
rule ga_sample_edges:
    output: trip=j(GA, "triplets.parquet"), pool=j(GA, "pool_text.parquet"),
    resources: mem_gb=120,
    shell: f"python {SCRIPTS}/sample_edges.py"

rule ga_pool_text:
    input: trip=GA_TRIPLETS,
    output: pool=GA_POOL_TEXT,
    resources: mem_gb=60,
    shell: f"python {SCRIPTS}/pool_text_from_triplets.py --triplets {{input.trip}} --out {{output.pool}}"

# fresh doc2lora embedding of the pool, per encoder
rule ga_embed_pool:
    input: pool=GA_POOL_TEXT,
    output: npz=j(GA, f"pool_genes_{{enc}}{GA_SAMPLE}.npz"),
    params: batch=lambda w: "16" if w.enc == "mistral" else "32",
    resources: gpu=1, mem_gb=40,
    shell: f"{GA_ENV} MAX_TOKENS=512 BATCH={{params.batch}} POOL_TEXT={{input.pool}} POOL_OUT={{output.npz}} {GPU} "
           f"python {SCRIPTS}/embed_pool.py {{wildcards.enc}}"

# Named for the sample it was trained on, because the bare name is a trap: on the
# machine that produced the paper, adapter_general_qwen.pt is byte-identical to the 2x
# adapter, and applying it reproduces the published vectors at cos .005 instead of 1.0.
rule ga_train:
    input: trip=GA_TRIPLETS, pool=j(GA, f"pool_genes_{{enc}}{GA_SAMPLE}.npz"),
    output: adapter=j(GA, f"adapter_general_{{enc}}{GA_SAMPLE}.pt"),
    resources: gpu=1, mem_gb=40,
    shell: f"{GA_ENV} TRIPLETS={{input.trip}} ADAPTER_OUT={{output.adapter}} {GPU} "
           f"python {SCRIPTS}/train_general.py {{wildcards.enc}}"


# ---- apply general adapter to each benchmark's genes ----
rule ga_apply_oax:
    input: adapter=j(GA, f"adapter_general_{{enc}}{GA_SAMPLE}.pt"), gene=j(OAX_EMB2, "{enc}_norm_lora_emb.npz"),
    output: gk=j(OAX_EMB2, "{enc}_genkron_emb.npz"),
    resources: gpu=1, mem_gb=60,
    shell: f"{GA_ENV} ADAPTER={{input.adapter}} {GPU} python {SCRIPTS}/apply_general.py {{wildcards.enc}} {{input.gene}} {{output.gk}} embeddings"

rule ga_apply_aps:
    input: adapter=j(GA, f"adapter_general_{{enc}}{GA_SAMPLE}.pt"), gene=j(APS_EMB2, "{enc}_norm_lora_emb.npz"),
    output: gk=j(APS_EMB2, "{enc}_genkron_emb.npz"),
    resources: gpu=1, mem_gb=60,
    wildcard_constraints: enc="|".join(GA_APS_ENC),
    shell: f"{GA_ENV} ADAPTER={{input.adapter}} {GPU} python {SCRIPTS}/apply_general.py {{wildcards.enc}} {{input.gene}} {{output.gk}} embeddings"

rule ga_apply_s2and:
    input: adapter=j(GA, f"adapter_general_{{enc}}{GA_SAMPLE}.pt"), gene=j(S2, "proc", "{ds}", "genes_{enc}.npz"),
    output: gk=j(S2, "proc", "{ds}", "genes_{enc}_genkron.npz"),
    resources: gpu=1, mem_gb=30,
    shell: f"{GA_ENV} ADAPTER={{input.adapter}} {GPU} python {SCRIPTS}/apply_general.py {{wildcards.enc}} {{input.gene}} {{output.gk}} embeddings"


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
