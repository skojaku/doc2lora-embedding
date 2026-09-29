# Group C similarity re-runs: modern encoders, the transformed-baseline control (App. I),
# and temporal hardening (App. A).  ONE chain, because all three write the same kind of row.
#
# Why a separate pool directory: adding methods changes the coverage intersection that defines the
# evaluation units, so folding them into data/uncertainty/pools would silently move the
# manuscript's Tab. 2 numbers.  Everything here lands in data/groupc/bench/, and Tab. 2 keeps its
# byte-identical inputs.
#
#   gcb_subset       -> subset_text.parquet     the papers the harness touches (bench.smk's eval-id lists)
#   gcb_embed        -> emb_{method}.npz        BGE / GTE / EmbeddingGemma (/ E5-Mistral) on the subset
#   gcb_pool_embed   -> pool_{method}.npz       the g_theta TRAINING pool in each baseline's space
#   gcb_train        -> adapter_{method}[_pre2018].pt   same triplets, same loss, same steps
#   gcb_apply        -> {method}_kron_gc.npz    the transform applied to each benchmark space
#   gcb_pool_scores  -> pools/{task}_{field}.parquet    paired per-unit scores, all methods
#   gcb_bootstrap    -> figs/groupc_similarity.tex + figs/groupc_temporal.tex
#
# RUN: snakemake groupc_bench -j3 --rerun-triggers mtime
import os
from os.path import join as j

FIGS_DIR = config.get("figs_dir", "figs")
GCB_DIR = j("data", "groupc", "bench")
GCB_POOLS = j(GCB_DIR, "pools")
BENCH_ROOT = config.get("bench_root", j("data", "bench"))
ICAE_BENCH = "data/icae"
GA_DIR = "data/general_adapter"
EMGEMMA_VENV = ".venv-emgemma"

GCB_FIELDS = config.get("gcb_fields", ["aps", "economics", "psychology"])
# encoders that have to be run here (no cached vectors for the benchmark subsets)
GCB_NEW_ENC = config.get("gcb_new_encoders", ["bge", "gte", "embeddinggemma"])
# every space that gets the SAME citation transform (the symmetric control + the paper's own gene)
GCB_TRANSFORM = config.get("gcb_transform_methods",
                           ["gene", "sbert", "specter2", "instructor", "embeddinggemma", "bge", "gte"])
# the temporally cut-off transform is expensive to add everywhere; temporal hardening needs the paper's transform
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
    "icae": ("icae_emb.npz", "vecs"),                 # the ICAE baseline, free to carry along
    # ICAE WITH the citation transform. Trained by the ICAE chain (icae.smk), not by gcb_train: the
    # ICAE space factorises over 128 memory tokens x 4,096 channels, which gcb_train's flat path does
    # not cover. Same citation supervision, a 16.8M-parameter map.
    "icae_genkron": ("icae_genkron_emb.npz", "vecs"),
}


def _field_text(field):
    return (j(APS_DIR, "paper_text.parquet") if field == "aps"
            else j(DATA_DIR, "fields", field, "paper_text.parquet"))


# Baselines whose FULL-corpus vectors are already on disk: the benchmark subset is a slice of them,
# not a fresh GPU pass. GTE is Tab. 2's own `baseline_gte.npz` (the sentence-transformers load), not
# the CLS-pooled `baseline_gte_large.npz` that t2l.smk uses for Text-to-LoRA's coordinates -- the
# control has to adapt the same vectors the reported row scores.
GCB_FULL_SRC = {
    "gte": lambda field: (j(APS_DIR, "embeddings", "baseline_gte.npz") if field == "aps"
                          else j(DATA_DIR, "fields", field, "embeddings", "baseline_gte.npz")),
}


# Which directory holds a field's RAW vectors. The manuscript's own pools score the FULL-corpus
# files, and for APS that matters: the corpus carries 644,022 papers while the benchmark subset
# carries 164,160, and collaboration prediction averages an author's window papers, so a narrower
# file yields a different author centroid (9.2% of those papers are outside the subset). The
# vectors themselves are identical row for row -- only the set of rows differs. Economics and
# Psychology are unaffected, because their embedding files never covered more than the subset.
# Their full-corpus copies have since been deleted, so they read the subset here.
GCB_FULL_FIELD_DIR = config.get("gcb_full_field_dir", {"aps": j(DATA_DIR, "aps", "embeddings")})
GCB_FULL_NEW = {"gte": "baseline_gte.npz", "embeddinggemma": "baseline_embeddinggemma.npz"}


def _field_emb_dir(field):
    return GCB_FULL_FIELD_DIR.get(field, j(BENCH_ROOT, field, "embeddings"))


def _subset_marker(field):
    """bench.smk's marker for "the benchmark subset for this field has been sliced".

    Fields whose vectors are read from a full corpus directory (APS) are not sliced, so
    they have nothing to wait for.
    """
    if field in GCB_FULL_FIELD_DIR:
        return []
    return j(BENCH_ROOT, field, "embeddings", ".sliced")


def _src_npz(field, method):
    """Where a method's RAW benchmark vectors live."""
    if method in GCB_CACHED:
        return j(_field_emb_dir(field), GCB_CACHED[method][0])
    # Encoders run by this chain. Where a full-corpus file exists for the field we read it, so the
    # row matches the one the manuscript scores; elsewhere we read this chain's own subset pass.
    # GTE means the sentence-transformers load (baseline_gte.npz), NOT the CLS-pooled
    # baseline_gte_large.npz that t2l.smk uses for Text-to-LoRA's coordinates. APS EmbeddingGemma was
    # re-embedded over the corpus on 2026-09-23; it had been missing since before this control ran.
    full = j(_field_emb_dir(field), GCB_FULL_NEW.get(method, ""))
    if field in GCB_FULL_FIELD_DIR and method in GCB_FULL_NEW and os.path.exists(full):
        return full
    return j(GCB_DIR, field, f"emb_{method}.npz")


# GOTCHA: triplets.parquet is NOT the sample the manuscript reports. The 2026-06-25
# data-scaling test overwrote it with a FRESH 2x resample (85,635 tuples over a 335,085-paper pool),
# and that resample is not a superset -- it shares only 82,635 of the 167,111 papers the reported
# transform was trained on, and just 29 of the 42,332 reported tuples survive inside it. The
# reported g_theta (adapter_general_qwen_1x.pt) and the ICAE transform both read triplets_1x.
# Every transform in this control therefore trains on the 1x sample, so the arms differ only in
# the space they act on.
GCB_TRIPLETS = config.get("gcb_triplets", GA_TRIPLETS)   # one definition, in general_adapter.smk
GCB_POOL_TEXT = config.get("gcb_pool_text", GA_POOL_TEXT)
GCB_POOL_SUFFIX = config.get("gcb_pool_suffix", "_1x")


def _pool_npz(method):
    """Where the g_theta TRAINING pool lives for that space, matching GCB_TRIPLETS."""
    if method == "gene":
        return j(GA_DIR, f"pool_genes_qwen{GCB_POOL_SUFFIX}.npz")
    return j(GCB_DIR, "pool", f"pool_{method}{GCB_POOL_SUFFIX}.npz")


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
        "set -a; source .env 2>/dev/null || true; set +a; "
        "NEED_MB=20000 bash workflow/scripts/gpu_lease.sh "
        "{params.py} workflow/scripts/groupc/gcb_embed.py --input {input.subset} "
        "--method {wildcards.method} --out {output.npz} --batch-size {params.batch}"


ruleorder: gcb_slice_cached > gcb_embed


rule gcb_slice_cached:
    input:
        src=lambda w: ancient(GCB_FULL_SRC[w.method](w.field)),
        subset=j(GCB_DIR, "{field}", "subset_text.parquet"),
    output:
        npz=j(GCB_DIR, "{field}", "emb_{method}.npz"),
    wildcard_constraints:
        field="|".join(GCB_FIELDS),
        method="|".join(GCB_FULL_SRC),
    resources:
        mem_gb=40,
    shell:
        "python workflow/scripts/groupc/gcb_slice.py --src {input.src} "
        "--subset {input.subset} --out {output.npz}"


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
        "set -a; source .env 2>/dev/null || true; set +a; "
        "NEED_MB=20000 bash workflow/scripts/gpu_lease.sh "
        "{params.py} workflow/scripts/groupc/gcb_embed.py --input {input.pool} "
        "--method {wildcards.method} --out {output.npz} --batch-size {params.batch}"


# The 1x pool is half-covered by the 2x pool, so only the missing papers get a GPU pass.
rule gcb_pool_embed_1x:
    input:
        pool=ancient(GCB_POOL_TEXT),
        existing=j(GCB_DIR, "pool", "pool_{method}.npz"),
    output:
        npz=j(GCB_DIR, "pool", "pool_{method}_1x.npz"),
    params:
        py=lambda w: _py(w.method),
        batch=lambda w: 24 if w.method == "gte" else 128,
    wildcard_constraints:
        method="|".join([m for m in GCB_TRANSFORM if m != "gene"] + ["specter1"]),
    resources:
        gpu=1,
        mem_gb=30,
    shell:
        "set -a; source .env 2>/dev/null || true; set +a; "
        "NEED_MB=20000 bash workflow/scripts/gpu_lease.sh "
        "{params.py} workflow/scripts/groupc/gcb_pool_incremental.py --method {wildcards.method} "
        "--pool {input.pool} --existing {input.existing} --out {output.npz} "
        "--batch-size {params.batch}"


rule gcb_train:
    input:
        pool=lambda w: ancient(_pool_npz(w.method)),
        triplets=ancient(GCB_TRIPLETS),
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


# `src` is a file inside the sliced benchmark subset, and bench.smk models that slice
# with a MARKER, because which files it writes depends on which embeddings exist. So the
# dependency is declared on the marker and the vector path travels as a param; declaring
# the npz directly leaves it with no producer on a clean tree.
rule gcb_apply:
    input:
        adapter=j(GCB_DIR, "transforms", "adapter_{method}{variant}.pt"),
        marker=lambda w: _subset_marker(w.field),
    params:
        src=lambda w: _src_npz(w.field, w.method),
    output:
        npz=j(GCB_DIR, "{field}", "{method}_kron_gc{variant}.npz"),
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
        out[m] = [j(_field_emb_dir(field), fn), key]
    for m in GCB_NEW_ENC:
        out[m] = [_src_npz(field, m), "vecs"]
    for m in GCB_TRANSFORM:
        out[f"{m}_kron_gc"] = [j(GCB_DIR, field, f"{m}_kron_gc.npz"), "vecs"]
    for m in GCB_TEMPORAL:
        out[f"{m}_kron_gc_pre2018"] = [j(GCB_DIR, field, f"{m}_kron_gc_pre2018.npz"), "vecs"]
    return out


def _score_inputs(w):
    ins = {"subset": j(GCB_DIR, w.field, "subset_text.parquet")}
    for m in GCB_NEW_ENC:
        ins[f"emb_{m}"] = _src_npz(w.field, m)
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
