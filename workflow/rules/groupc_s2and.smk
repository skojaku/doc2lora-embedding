# Symmetric-supervision control, S2AND half + the merged 14-test table (#145).
#
# groupc_bench.smk gives every FIELD-task space the same citation-trained bijective transform
# (#93's control, 9 tests). The other five tests of the reported table are S2AND name
# disambiguation, and they were left asymmetric for one concrete reason: S2AND scores against the
# SPECTER v1 vectors the benchmark ships, and no transform existed in that space. This file builds
# it and finishes the control.
#
#   gcb_pool_embed_specter1 -> pool/pool_specter1.npz       the g_theta pool in SPECTER v1 space
#   gcb_train_specter1      -> transforms/adapter_specter1.pt
#   gcb_s2and_apply         -> s2and/{ds}/{method}_kron_gc.npz
#   gcb_s2and_pool          -> pools/s2and_{ds}.parquet     per-signature B3 (P,R), all methods
#   gcb_s2and_bootstrap     -> gcb_s2and_summary.csv
#   gcb_symmetric           -> figs/groupc_similarity_symmetric.tex + gcb_symmetric.csv (all 14)
#
# Adapted vectors carry the paper ids of their source, so adding them cannot move the coverage
# intersection that defines the evaluation units -- the reported numbers stay where they are.
#
# RUN: snakemake groupc_s2and -j3 --rerun-triggers mtime
from os.path import join as j

GCB_S2AND = config.get("gcb_s2and", ["zbmath", "qian", "arnetminer", "pubmed", "kisti"])
GCB_S2AND_ENC = config.get("gcb_s2and_enc", "qwen")
S2_PROC = "data/s2and/proc"
GCB_S2_DIR = j(GCB_DIR, "s2and")

# space -> the npz S2AND scores for it.  `specter` is the benchmark's own precomputed feature.
GCB_S2_SRC = {
    "gene": f"genes_{GCB_S2AND_ENC}.npz",
    "specter": "specter.npz",
    "sbert": "sbert.npz",
    "instructor": "instructor.npz",
    "embeddinggemma": "embeddinggemma.npz",
    "gte": "gte.npz",
}
# space -> the transform trained in that space (SPECTER v1 gets its own; the rest are shared with
# the field arm, which is the point of the control)
GCB_S2_ADAPTER = {"gene": "gene", "specter": "specter1", "sbert": "sbert",
                  "instructor": "instructor", "embeddinggemma": "embeddinggemma", "gte": "gte"}


rule gcb_pool_embed_specter1:
    input:
        pool=ancient(j(GA_DIR, "pool_text.parquet")),
    output:
        npz=j(GCB_DIR, "pool", "pool_specter1.npz"),
    resources:
        gpu=1,
        mem_gb=30,
    shell:
        "set -a; source .env 2>/dev/null; set +a; "
        "NEED_MB=20000 bash workflow/scripts/gpu_lease.sh "
        "python workflow/scripts/groupc/gcb_embed.py --input {input.pool} "
        "--method specter1 --out {output.npz} --batch-size 128"


rule gcb_train_specter1:
    input:
        pool=j(GCB_DIR, "pool", f"pool_specter1{GCB_POOL_SUFFIX}.npz"),
        triplets=ancient(GCB_TRIPLETS),
    output:
        adapter=j(GCB_DIR, "transforms", "adapter_specter1.pt"),
        meta=j(GCB_DIR, "transforms", "adapter_specter1.pt.json"),
    params:
        steps=GCB_STEPS,
    resources:
        gpu=1,
        mem_gb=30,
    shell:
        "NEED_MB=6000 bash workflow/scripts/gpu_lease.sh "
        "python workflow/scripts/groupc/gcb_train_transform.py --pool {input.pool} "
        "--triplets {input.triplets} --out {output.adapter} --layers 1 --steps {params.steps}"


rule gcb_s2and_apply:
    input:
        adapter=lambda w: j(GCB_DIR, "transforms", f"adapter_{GCB_S2_ADAPTER[w.method]}.pt"),
        src=lambda w: ancient(j(S2_PROC, w.ds, GCB_S2_SRC[w.method])),
    output:
        npz=j(GCB_S2_DIR, "{ds}", "{method}_kron_gc.npz"),
    wildcard_constraints:
        ds="|".join(GCB_S2AND),
        method="|".join(GCB_S2_SRC),
    resources:
        gpu=1,
        mem_gb=20,
    shell:
        "NEED_MB=6000 bash workflow/scripts/gpu_lease.sh "
        "python workflow/scripts/groupc/gcb_apply.py --adapter {input.adapter} "
        "--in {input.src} --out {output.npz}"


rule gcb_s2and_pool:
    input:
        adapted=expand(j(GCB_S2_DIR, "{{ds}}", "{method}_kron_gc.npz"), method=list(GCB_S2_SRC)),
        sig=ancient(j(S2_PROC, "{ds}", "sig_table.parquet")),
    output:
        parquet=j(GCB_POOLS, "s2and_{ds}.parquet"),
    params:
        enc=GCB_S2AND_ENC,
    wildcard_constraints:
        ds="|".join(GCB_S2AND),
    resources:
        mem_gb=40,
    shell:
        "python workflow/scripts/groupc/gcb_s2and_pool.py --ds {wildcards.ds} "
        "--enc {params.enc} --out {output.parquet}"


rule gcb_s2and_bootstrap:
    input:
        pools=expand(j(GCB_POOLS, "s2and_{ds}.parquet"), ds=GCB_S2AND),
    output:
        csv=j(GCB_DIR, "gcb_s2and_summary.csv"),
    params:
        pool_dir=GCB_POOLS,
        nboot=GCB_NBOOT,
    resources:
        mem_gb=20,
    script:
        "../scripts/groupc/gcb_s2and_bootstrap.py"


rule gcb_symmetric:
    input:
        fields=j(GCB_DIR, "gcb_summary.csv"),
        s2and=j(GCB_DIR, "gcb_s2and_summary.csv"),
    output:
        table=j(FIGS_DIR, "groupc_similarity_symmetric.tex"),
        values_table=j(FIGS_DIR, "symmetric_adapter_values.tex"),
        csv=j(GCB_DIR, "gcb_symmetric.csv"),
    params:
        fields=GCB_FIELDS,
        s2and=GCB_S2AND,
    resources:
        mem_gb=8,
    script:
        "../scripts/groupc/gcb_symmetric_table.py"


# Appendix dot plot of the same scores (replaces tab:symmetric-values): per space, the mean log
# gain ln(with g_theta / without), per task and over all 14.
rule fig_symmetric_gain:
    input:
        fields=j(GCB_DIR, "gcb_summary.csv"),
        s2and=j(GCB_DIR, "gcb_s2and_summary.csv"),
    output:
        pdf=j(FIGS_DIR, "symmetric_adapter_gain.pdf"),
    resources:
        mem_gb=4,
    script:
        "../plot/fig_symmetric_gain.py"


# The figure shows gains only, so the raw level of each space (ICAE lowest) gets a short table.
rule symmetric_raw_table:
    input:
        fields=j(GCB_DIR, "gcb_summary.csv"),
        s2and=j(GCB_DIR, "gcb_s2and_summary.csv"),
    output:
        table=j(FIGS_DIR, "symmetric_raw_scores.tex"),
    resources:
        mem_gb=2,
    script:
        "../scripts/groupc/gcb_raw_table.py"


# ── paired head-to-head (#145): is the difference between two rows real? ────────────────
# gcb_bootstrap resamples each method on its own, so its intervals answer "how precise is this
# number". Two rows scored on the same units are correlated, and the reader's question is whether
# A is above B. These rules resample the units ONCE per replicate and recompute A - B on them.
GCB_H2H_SPACES = config.get("gcb_h2h_spaces",
                            ["gene", "sbert", "specter2", "instructor", "embeddinggemma", "gte",
                             "icae"])
# field-pool column -> S2AND method name (they differ for the gene and SPECTER rows)
GCB_H2H_S2 = {"gene": ("genkron", "gene"), "specter2": ("specter_kron_gc", "specter"),
              "icae": ("icae_genkron", "icae")}


def _h2h_names(space):
    """(adapted, raw) column in the field pools, and the same pair in the S2AND pools."""
    # ICAE's transform is trained by the ICAE chain (icae.smk) under its own column name
    adapted = {"gene": "genkron", "icae": "icae_genkron"}.get(space, f"{space}_kron_gc")
    s2 = GCB_H2H_S2.get(space, (f"{space}_kron_gc", space))
    return adapted, space, s2[0], s2[1]


rule gcb_headtohead:
    input:
        pools=expand(j(GCB_POOLS, "s2and_{ds}.parquet"), ds=GCB_S2AND),
        # gcb_pool_scores writes a marker, not one file per task, so the dependency is
        # declared on the marker; the parquets it drops beside it are read at run time.
        field=expand(j(GCB_POOLS, "{field}_pools.json"), field=GCB_FIELDS),
    output:
        csv=j(GCB_DIR, "h2h_{space}_adapted_vs_raw.csv"),
    params:
        names=lambda w: _h2h_names(w.space),
        nboot=config.get("gcb_h2h_nboot", 400),
    wildcard_constraints:
        space="|".join(GCB_H2H_SPACES),
    resources:
        mem_gb=40,
    shell:
        "python workflow/scripts/groupc/gcb_headtohead.py --a {params.names[0]} "
        "--b {params.names[1]} --a-s2and {params.names[2]} --b-s2and {params.names[3]} "
        "--nboot {params.nboot} --out {output.csv}"


# \doctolora against ICAE, both carrying the transform: the two decodable compressed spaces, the
# comparison the retrieval table cannot settle by eye because the rows sit on top of each other.
rule gcb_headtohead_icae:
    input:
        pools=expand(j(GCB_POOLS, "s2and_{ds}.parquet"), ds=GCB_S2AND),
        # gcb_pool_scores writes a marker, not one file per task, so the dependency is
        # declared on the marker; the parquets it drops beside it are read at run time.
        field=expand(j(GCB_POOLS, "{field}_pools.json"), field=GCB_FIELDS),
    output:
        csv=j(GCB_DIR, "gcb_headtohead_d2l_vs_icae.csv"),
    params:
        nboot=config.get("gcb_h2h_nboot", 400),
    resources:
        mem_gb=40,
    shell:
        "python workflow/scripts/groupc/gcb_headtohead.py --a genkron --b icae_genkron "
        "--nboot {params.nboot} --out {output.csv}"


rule gcb_headtohead_table:
    input:
        pairs=expand(j(GCB_DIR, "h2h_{space}_adapted_vs_raw.csv"), space=GCB_H2H_SPACES),
    output:
        table=j(FIGS_DIR, "symmetric_adapter_summary.tex"),
    resources:
        mem_gb=8,
    script:
        "../scripts/groupc/gcb_headtohead_table.py"


# SI companion to Table 1: the collaboration AUC of each anchor window, from the manuscript's own
# pools. Table 1 pools the windows into one AUC (the estimator App. B describes as of a0a7146c), and
# this table shows the windows behind it.
rule collab_per_window:
    input:
        pools=expand(j(POOLS, "collab_{field}_{enc}.parquet"),
                     field=GCB_FIELDS, enc=[config.get("unc_enc", "qwen")]),
    output:
        table=j(FIGS_DIR, "collab_per_window.tex"),
    params:
        pools=POOLS,
        fields=GCB_FIELDS,
        enc=config.get("unc_enc", "qwen"),
    resources:
        mem_gb=16,
    script:
        "../scripts/groupc/collab_per_window.py"


rule groupc_s2and:
    input:
        j(GCB_DIR, "gcb_s2and_summary.csv"),
        j(FIGS_DIR, "groupc_similarity_symmetric.tex"),
        j(FIGS_DIR, "symmetric_adapter_values.tex"),
        j(FIGS_DIR, "symmetric_adapter_gain.pdf"),
        j(FIGS_DIR, "symmetric_raw_scores.tex"),
        j(FIGS_DIR, "symmetric_adapter_summary.tex"),
        j(FIGS_DIR, "collab_per_window.tex"),
        j(GCB_DIR, "gcb_headtohead_d2l_vs_icae.csv"),
