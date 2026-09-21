# Group C similarity re-runs: modern encoders (#69), the transformed-baseline control (#93),
# and temporal hardening (#72).  ONE chain, because all three write the same kind of row.
#
# Why a separate pool directory: adding methods changes the coverage intersection that defines the
# evaluation units, so folding them into exps/2026-06-10-uncertainty/pools would silently move the
# manuscript's Table 1 numbers.  Everything here lands in data/groupc/bench/, and Table 1 keeps its
# byte-identical inputs.
#
#   gcb_subset       -> subset_text.parquet     the papers the harness touches (#63's eval-id lists)
#   gcb_embed        -> emb_{method}.npz        BGE / GTE / EmbeddingGemma (/ E5-Mistral) on the subset
#   gcb_pool_embed   -> pool_{method}.npz       the g_theta TRAINING pool in each baseline's space
#   gcb_train        -> adapter_{method}[_pre2018].pt   same triplets, same loss, same steps
#   gcb_apply        -> {method}_kron_gc.npz    the transform applied to each benchmark space
#   gcb_pool_scores  -> pools/{task}_{field}.parquet    paired per-unit scores, all methods
#   gcb_bootstrap    -> figs/groupc_similarity.tex (#69/#93) + figs/groupc_temporal.tex (#72)
#
# RUN: snakemake groupc_bench -j3 --rerun-triggers mtime
from os.path import join as j

FIGS_DIR = config.get("figs_dir", "figs")
GCB_DIR = j("data", "groupc", "bench")
GCB_POOLS = j(GCB_DIR, "pools")
BENCH_ROOT = config.get("bench_root", j("data", "bench"))
ICAE_BENCH = "exps/2026-06-21-icae-benchmark"
GA_DIR = "exps/2026-06-10-general-adapter"
EMGEMMA_VENV = ".venv-emgemma"

GCB_FIELDS = config.get("gcb_fields", ["aps", "economics", "psychology"])
# encoders that have to be run here (no cached vectors for the benchmark subsets)
GCB_NEW_ENC = config.get("gcb_new_encoders", ["bge", "gte", "embeddinggemma"])
# every space that gets the SAME citation transform (the #93 control + the paper's own gene)
GCB_TRANSFORM = config.get("gcb_transform_methods",
                           ["gene", "sbert", "specter2", "instructor", "embeddinggemma", "bge", "gte"])
# the temporally cut-off transform is expensive to add everywhere; #72 needs the paper's transform
# plus one text reference
GCB_TEMPORAL = config.get("gcb_temporal_methods", ["gene", "sbert"])
GCB_CUTOFF = config.get("gcb_cutoff_year", 2018)
GCB_TOPIC_SPLIT = config.get("gcb_topic_split_year", 2016)
GCB_TOPIC_N = config.get("gcb_topic_n", 60000)
GCB_NBOOT = config.get("gcb_nboot", 500)
GCB_STEPS = config.get("gcb_train_steps", 8000)
GCB_YEARS = config.get("openalex_master_paper_table",
                       "/data2/datasets/openalex/preprocessed/paper_table.csv")

# cached benchmark-subset vectors (produced by earlier chains; read-only here)
GCB_CACHED = {
    "gene": ("qwen_norm_lora_emb.npz", "embeddings"),
    "genkron": ("qwen_genkron_emb.npz", "embeddings"),
    "sbert": ("sbert_allmpnet.npz", "vecs"),
    "specter2": ("baseline_specter2.npz", "vecs"),
    "instructor": ("baseline_instructor.npz", "vecs"),
    "icae": ("icae_emb.npz", "vecs"),                 # #63's ICAE baseline, free to carry along
}


def _field_text(field):
    return (j(APS_DIR, "paper_text.parquet") if field == "aps"
            else j(DATA_DIR, "fields", field, "paper_text.parquet"))


def _src_npz(field, method):
    """Where a method's RAW benchmark vectors live."""
    if method in GCB_CACHED:
        return j(BENCH_ROOT, field, "embeddings", GCB_CACHED[method][0])
    return j(GCB_DIR, field, f"emb_{method}.npz")


def _slice_marker(field):
    """bench.smk's marker for "the benchmark subset for this field has been sliced"."""
    return j(BENCH_ROOT, field, "embeddings", ".sliced")


def _pool_npz(method):
    """Where the g_theta TRAINING pool lives for that space."""
    if method == "gene":
        return j(GA_DIR, "pool_genes_qwen.npz")
    return j(GCB_DIR, "pool", f"pool_{method}.npz")


def _py(method):
    return j(EMGEMMA_VENV, "bin", "python") if method == "embeddinggemma" else "python"


rule gcb_subset:
    input:
        paper_text=lambda w: ancient(_field_text(w.field)),
    output:
        subset=j(GCB_DIR, "{field}", "subset_text.parquet"),
    params:
        eval_ids=lambda w: j(ICAE_BENCH, f"{w.field}_eval_ids.parquet"),
    wildcard_constraints:
        field="|".join(GCB_FIELDS),
    resources:
        mem_gb=30,
    script:
        "../scripts/groupc/gcb_subset.py"


rule gcb_embed:
    input:
        subset=j(GCB_DIR, "{field}", "subset_text.parquet"),
    output:
        npz=j(GCB_DIR, "{field}", "emb_{method}.npz"),
    params:
        py=lambda w: _py(w.method),
        batch=lambda w: 8 if w.method == "e5mistral" else 128,
    wildcard_constraints:
        field="|".join(GCB_FIELDS),
        method="|".join(GCB_NEW_ENC + ["e5mistral"]),
    resources:
        gpu=1,
        mem_gb=30,
    shell:
        "set -a; source .env 2>/dev/null; set +a; "
        "NEED_MB=20000 bash workflow/scripts/gpu_lease.sh "
        "{params.py} workflow/scripts/groupc/gcb_embed.py --input {input.subset} "
        "--method {wildcards.method} --out {output.npz} --batch-size {params.batch}"


rule gcb_pool_embed:
    input:
        pool=ancient(j(GA_DIR, "pool_text.parquet")),
    output:
        npz=j(GCB_DIR, "pool", "pool_{method}.npz"),
    params:
        py=lambda w: _py(w.method),
        batch=lambda w: 8 if w.method == "e5mistral" else 128,
    wildcard_constraints:
        method="|".join([m for m in GCB_TRANSFORM if m != "gene"] + ["e5mistral"]),
    resources:
        gpu=1,
        mem_gb=30,
    shell:
        "set -a; source .env 2>/dev/null; set +a; "
        "NEED_MB=20000 bash workflow/scripts/gpu_lease.sh "
        "{params.py} workflow/scripts/groupc/gcb_embed.py --input {input.pool} "
        "--method {wildcards.method} --out {output.npz} --batch-size {params.batch}"


rule gcb_train:
    input:
        pool=lambda w: ancient(_pool_npz(w.method)),
        triplets=ancient(j(GA_DIR, "triplets.parquet")),
    output:
        adapter=j(GCB_DIR, "transforms", "adapter_{method}{variant}.pt"),
        meta=j(GCB_DIR, "transforms", "adapter_{method}{variant}.pt.json"),
    params:
        layers=lambda w: 36 if w.method == "gene" else 1,
        steps=GCB_STEPS,
        cutoff=lambda w: GCB_CUTOFF if w.variant else 0,
        years=GCB_YEARS,
    wildcard_constraints:
        method="|".join(GCB_TRANSFORM),
        variant="|_pre2018",
    resources:
        gpu=1,
        mem_gb=60,
    shell:
        "NEED_MB=6000 bash workflow/scripts/gpu_lease.sh "
        "python workflow/scripts/groupc/gcb_train_transform.py --pool {input.pool} "
        "--triplets {input.triplets} --out {output.adapter} --layers {params.layers} "
        "--steps {params.steps} --cutoff {params.cutoff} --year-table {params.years}"


rule gcb_apply:
    input:
        adapter=j(GCB_DIR, "transforms", "adapter_{method}{variant}.pt"),
        # The benchmark-subset vectors are side products of bench.smk's slicer,
        # which declares a marker rather than a file list (the set depends on
        # which embeddings exist). Depend on the marker and take the path as a
        # param, so this chain is reachable from a cold start.
        sliced=lambda w: _slice_marker(w.field),
    output:
        npz=j(GCB_DIR, "{field}", "{method}_kron_gc{variant}.npz"),
    params:
        src=lambda w: _src_npz(w.field, w.method),
    wildcard_constraints:
        field="|".join(GCB_FIELDS),
        method="|".join(GCB_TRANSFORM),
        variant="|_pre2018",
    resources:
        gpu=1,
        mem_gb=60,
    shell:
        "NEED_MB=6000 bash workflow/scripts/gpu_lease.sh "
        "python workflow/scripts/groupc/gcb_apply.py --adapter {input.adapter} "
        "--in {params.src} --out {output.npz}"


def _methods_for(field):
    """name -> [path, key] for every method that should appear in the pooled scores."""
    out = {}
    for m, (fn, key) in GCB_CACHED.items():
        out[m] = [j(BENCH_ROOT, field, "embeddings", fn), key]
    for m in GCB_NEW_ENC:
        out[m] = [j(GCB_DIR, field, f"emb_{m}.npz"), "vecs"]
    for m in GCB_TRANSFORM:
        out[f"{m}_kron_gc"] = [j(GCB_DIR, field, f"{m}_kron_gc.npz"), "vecs"]
    for m in GCB_TEMPORAL:
        out[f"{m}_kron_gc_pre2018"] = [j(GCB_DIR, field, f"{m}_kron_gc_pre2018.npz"), "vecs"]
    return out


def _score_inputs(w):
    ins = {"subset": j(GCB_DIR, w.field, "subset_text.parquet")}
    for m in GCB_NEW_ENC:
        ins[f"emb_{m}"] = j(GCB_DIR, w.field, f"emb_{m}.npz")
    for m in GCB_TRANSFORM:
        ins[f"k_{m}"] = j(GCB_DIR, w.field, f"{m}_kron_gc.npz")
    for m in GCB_TEMPORAL:
        ins[f"t_{m}"] = j(GCB_DIR, w.field, f"{m}_kron_gc_pre2018.npz")
    return ins


rule gcb_pool_scores:
    input:
        unpack(_score_inputs),
    output:
        done=j(GCB_POOLS, "{field}_pools.json"),
    params:
        methods=lambda w: _methods_for(w.field),
        out_dir=GCB_POOLS,
        topic_n=GCB_TOPIC_N,
        temporal_split=GCB_TOPIC_SPLIT,
        aps_table=config["aps_paper_table"],
    wildcard_constraints:
        field="|".join(GCB_FIELDS),
    resources:
        gpu=1,
        mem_gb=120,
    script:
        "../scripts/groupc/gcb_pool_scores.py"


rule gcb_bootstrap:
    input:
        pools=expand(j(GCB_POOLS, "{field}_pools.json"), field=GCB_FIELDS),
    output:
        csv=j(GCB_DIR, "gcb_summary.csv"),
        table=j(FIGS_DIR, "groupc_similarity.tex"),
        temporal_table=j(FIGS_DIR, "groupc_temporal.tex"),
    params:
        nboot=GCB_NBOOT,
        pool_dir=GCB_POOLS,
        fields=GCB_FIELDS,
    resources:
        mem_gb=60,
    script:
        "../scripts/groupc/gcb_bootstrap.py"


rule groupc_bench:
    input:
        j(GCB_DIR, "gcb_summary.csv"),
        j(FIGS_DIR, "groupc_similarity.tex"),
        j(FIGS_DIR, "groupc_temporal.tex"),
