# Benchmark uncertainty: per-unit score pools + bootstrap CIs (data/uncertainty).
#
# Every headline number (collab AUC / next-paper AUC / topic macro-F1 / S2AND B3-F1) is a single-seed
# point estimate. These rules dump the PAIRED per-unit evaluation scores (same units across all methods)
# so confidence intervals can be bootstrapped, then run the reference bootstrap:
#   unc_pool_field  : collab/next-paper/topic per-unit pools     -> pools/{collab,np,topic}_{field}_{enc}.parquet
#   unc_pool_s2and  : per-signature B3 (P,R) per method          -> pools/s2and_{ds}_{enc}.parquet
#   unc_bootstrap   : resample each pool N times, recompute metric-> uncertainty_summary.csv + bootstrap_replicates.parquet
#
# The pool scripts are CPU-only (CUDA_VISIBLE_DEVICES="" — embeddings are precomputed; s2and trains a tiny
# same-author Kron adapter on CPU). They consume the gene / kron / genkron / text embeddings produced by the
# `kron`, `general`, `fields`, `baselines`, and `s2and` sub-workflows, so run those first (or let the DAG
# build them). Like `kron`, pass --rerun-triggers mtime so Snakemake does not try to RE-EXTRACT the genes:
#   snakemake uncertainty --rerun-triggers mtime -j4
#
# Method coverage mirrors the reported matrix: economics/psychology carry a per-field citation adapter
# ({enc}_kron) AND the general adapter ({enc}_genkron); APS carries genkron for all encoders but a per-field
# kron only for mistral (gemma/qwen APS kron was never part of the matrix — score_pool_field skips it if absent).
from os.path import join as j

UNC = "data/uncertainty"
S2U = "data/s2and"
POOLS = j(UNC, "pools")

# field tasks (collab/next-paper/topic): encoders available per field
UNC_FIELD_ENC = config.get("unc_field_enc", {
    "economics": ["gemma", "qwen", "mistral"],
    "psychology": ["gemma", "qwen", "mistral"],
    "aps": ["gemma", "qwen", "mistral"],
})
# name disambiguation
UNC_S2AND = config.get("unc_s2and", ["zbmath", "qian", "arnetminer", "pubmed", "kisti"])
UNC_S2AND_ENC = config.get("unc_s2and_enc", ["gemma", "qwen", "mistral"])
UNC_NBOOT = config.get("unc_nboot", 1000)

# fields where a per-field citation Kron adapter ({enc}_kron_emb.npz) is part of the matrix
UNC_KRON_FIELDS = ("economics", "psychology")


def _field_pool_inputs(w):
    """Strictly-required inputs + the kron/genkron embeddings that score_pool_field.py opportunistically
    includes. kron is declared ONLY where it exists (econ/psych all-enc, aps-mistral) so the DAG never tries
    to build APS gemma/qwen kron. kron/text/collab/topics use DATA_DIR (./data, matches kron_adapter.smk);
    genkron uses plain 'data' (matches general_adapter.smk's ga_apply_* outputs)."""
    if w.field == "aps":
        emb_k = j(DATA_DIR, "aps", "embeddings")        # ./data/... (kron + text producers)
        emb_g = j("data", "aps", "embeddings")          # data/...   (genkron producer)
    else:
        emb_k = j(DATA_DIR, "fields", w.field, "embeddings")
        emb_g = j("data", "fields", w.field, "embeddings")
    ins = {
        "gene": j(emb_k, f"{w.enc}_norm_lora_emb.npz"),
        "genkron": j(emb_g, f"{w.enc}_genkron_emb.npz"),
        "sbert": j(emb_k, "sbert_allmpnet.npz"),
        "specter2": j(emb_k, "baseline_specter2.npz"),
        "instructor": j(emb_k, "baseline_instructor.npz"),
        "gte": j(emb_k, "baseline_gte.npz"),
        "collab": j(DATA_DIR, f"collab_scores_{w.field}_hard.parquet"),
    }
    if w.field in UNC_KRON_FIELDS or (w.field == "aps" and w.enc == "mistral"):
        ins["kron"] = j(emb_k, f"{w.enc}_kron_emb.npz")
    if w.field != "aps":
        ins["topics"] = j(DATA_DIR, "fields", w.field, "paper_topics.parquet")
    return ins


rule unc_pool_field:
    input:
        unpack(_field_pool_inputs),
    output:
        collab=j(POOLS, "collab_{field}_{enc}.parquet"),
        np=j(POOLS, "np_{field}_{enc}.parquet"),
        topic=j(POOLS, "topic_{field}_{enc}.parquet"),
    resources:
        mem_gb=60,
    wildcard_constraints:
        field="|".join(UNC_FIELD_ENC),
        enc="gemma|qwen|mistral",
    shell:
        f'CUDA_VISIBLE_DEVICES="" python {SCRIPTS}/score_pool_field.py {{wildcards.field}} {{wildcards.enc}}'


rule unc_pool_s2and:
    input:
        sig=j(S2U, "proc", "{ds}", "sig_table.parquet"),
        gene=j(S2U, "proc", "{ds}", "genes_{enc}.npz"),
        spec=j(S2U, "proc", "{ds}", "specter.npz"),
        sbert=j(S2U, "proc", "{ds}", "sbert.npz"),
        instr=j(S2U, "proc", "{ds}", "instructor.npz"),
        gte=j(S2U, "proc", "{ds}", "gte.npz"),
        genkron=j(S2U, "proc", "{ds}", "genes_{enc}_genkron.npz"),
    output:
        parquet=j(POOLS, "s2and_{ds}_{enc}.parquet"),
    resources:
        mem_gb=40,
    wildcard_constraints:
        ds="|".join(UNC_S2AND),
        enc="|".join(UNC_S2AND_ENC),
    shell:
        f'CUDA_VISIBLE_DEVICES="" python {SCRIPTS}/score_pool_s2and.py {{wildcards.ds}} {{wildcards.enc}}'


def _unc_pool_targets():
    out = []
    for f, encs in UNC_FIELD_ENC.items():
        for e in encs:
            out += [j(POOLS, f"collab_{f}_{e}.parquet"),
                    j(POOLS, f"np_{f}_{e}.parquet"),
                    j(POOLS, f"topic_{f}_{e}.parquet")]
    for ds in UNC_S2AND:
        for e in UNC_S2AND_ENC:
            out.append(j(POOLS, f"s2and_{ds}_{e}.parquet"))
    return out


rule unc_bootstrap:
    input:
        _unc_pool_targets(),
    output:
        summary=j(UNC, "uncertainty_summary.csv"),
        reps=j(UNC, "bootstrap_replicates.parquet"),
    params:
        nboot=UNC_NBOOT,
    resources:
        mem_gb=40,
    shell:
        f"python {SCRIPTS}/bootstrap.py {{params.nboot}}"


# LaTeX tables from the bootstrap summary (\input-ed by the manuscript):
# tab:similarity (main text) and tab:encoder-matrix (appendix).
TAB_SIMILARITY = j(FIGS_DIR, "similarity_benchmarks.tex")
TAB_TASK_TRANSFORM = j(FIGS_DIR, "similarity_task_transform.tex")
TAB_ENCODER_MATRIX = j(FIGS_DIR, "encoder_matrix.tex")

# Encoder used throughout the benchmark tables: Qwen3-4B for the field tasks and
# for every S2AND dataset (one encoder across the whole benchmark to simplify the
# story; per-encoder field comparison lives in tab:encoder-matrix).
UNC_TABLE_ENC = config.get("unc_table_enc", "qwen")
UNC_S2AND_BEST_ENC = config.get("unc_s2and_best_enc", {
    ds: "qwen" for ds in ("zbmath", "qian", "arnetminer", "pubmed", "kisti")
})


rule tab_similarity_benchmarks:
    input:
        summary=rules.unc_bootstrap.output.summary,
    output:
        similarity=TAB_SIMILARITY,
        task_transform=TAB_TASK_TRANSFORM,
        encoder_matrix=TAB_ENCODER_MATRIX,
    params:
        main_enc=UNC_TABLE_ENC,
        s2and_enc=UNC_S2AND_BEST_ENC,
    script:
        "../plot/tab_similarity_benchmarks.py"


FIG_SIMILARITY = j(FIGS_DIR, "similarity_benchmarks.pdf")

rule fig_similarity_benchmarks:
    input:
        summary=rules.unc_bootstrap.output.summary,
    output:
        pdf=FIG_SIMILARITY,
    script:
        "../plot/fig_similarity_benchmarks.py"


rule uncertainty_all:
    input:
        rules.unc_bootstrap.output,
        rules.tab_similarity_benchmarks.output,
        rules.fig_similarity_benchmarks.output,
