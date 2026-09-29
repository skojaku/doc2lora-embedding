# Incoherent-cluster / norm-matched control for cluster labelling (#101).
#
# Sec. 4.3 says the NORM of a vector sets the abstraction level of its decode, and a centroid's norm
# is set by how mutually aligned its members are.  If that is the whole story, a cluster of unrelated
# papers also decodes to a confident broad field name, and Sec. 4.2 would be measuring dispersion
# rather than shared content.  This chain runs the missing control:
#
#   incoh_build   -> clusters.json          matched-cardinality cross-PACS-chapter clusters (2 draws)
#   incoh_means   -> means_{arm}_s{i}.npz   full-rank Doc2LoRA centroid per cluster, GPU-sharded
#   incoh_decode  -> incoh_labels.json      same 2-3 word field prompt: real / control / norm-matched
#   incoh_score   -> incoh_scores.json + incoh_rows.parquet (SBERT + judge panel)
#   incoh_report  -> figs/incoherent_control.{tex,pdf} (CPU, from the two files above)
#
# The real arm is re-extracted here (not read from data/labels) so both arms go
# through one code path; the manuscript's own label artifacts are left untouched.
#
# RUN: snakemake groupc_incoherent -j4 --rerun-triggers mtime
from os.path import join as j

FIGS_DIR = config.get("figs_dir", "figs")
INC_DIR = j("data", "groupc", "incoherent")
INC_SHARDS = config.get("incoh_shards", 4)
INC_DRAWS = config.get("incoh_draws", 2)
INC_SEED = config.get("incoh_seed", 0)
INC_TITLES = config.get("incoh_n_titles", 12)
INC_SBERT_N = config.get("incoh_n_members_sbert", 200)
INC_JUDGES = config.get("incoh_use_judges", True)
INC_SBERT_MODEL = config.get("compat_sbert_model", "sentence-transformers/all-mpnet-base-v2")

BT = "data/labels"
CA = "data/pacs/results"

INC_CLUSTERS = j(INC_DIR, "clusters.json")
INC_LABELS = j(INC_DIR, "incoh_labels.json")
INC_SCORES = j(INC_DIR, "incoh_scores.json")
INC_ROWS = j(INC_DIR, "incoh_rows.parquet")
INC_TABLE = j(FIGS_DIR, "incoherent_control.tex")
INC_FIG = j(FIGS_DIR, "incoherent_control.pdf")

# Shared preamble for the Doc2LoRA GPU scripts: HF token, the ctx_to_lora source tree (the editable
# install's .pth is stale), and the Qwen3-4B hypernetwork checkpoint.
D2L_ENV = (
    "set -a; source .env 2>/dev/null || true; set +a; "
    f"export DOC_TO_LORA_SRC={config.get('doc_to_lora_src', 'doc-to-lora/src')} "
    f"PYTHONPATH={config.get('doc_to_lora_src', 'doc-to-lora/src')} "
    f"HF_HOME={config.get('hf_home', 'data/agent_assets/hf_cache')} "
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; "
)


rule incoh_build:
    input:
        paper_groups=ancient(j(CA, "paper_groups.parquet")),
        paper_text=ancient(j(APS_DIR, "paper_text.parquet")),
    output:
        clusters=INC_CLUSTERS,
    params:
        # Frozen artifact of the already-run `baseline_trees` chain: passed as a param, not an input,
        # so this experiment can never re-trigger (and overwrite) the manuscript's label pipeline.
        means=j(BT, "qwen_fullrank_means.npz"),
        n_draws=INC_DRAWS,
        seed=INC_SEED,
    resources:
        mem_gb=16,
    script:
        "../scripts/groupc/incoh_build.py"


rule incoh_means:
    input:
        clusters=INC_CLUSTERS,
        ckpt=ancient(QWEN_CHECKPOINT_PATH),
    output:
        npz=j(INC_DIR, "means_{arm}_s{i}.npz"),
    params:
        nshards=INC_SHARDS,
        ckpt=QWEN_CHECKPOINT_PATH,
    wildcard_constraints:
        arm="real|control",
    resources:
        gpu=1,
        mem_gb=30,
    # Other users park resident services on these cards and grow into them mid-run, so lease only a
    # roomy GPU, keep the per-batch token budget modest, and retry rather than fail the whole chain.
    retries: 3
    shell:
        D2L_ENV +
        "NEED_MB=26000 bash workflow/scripts/gpu_lease.sh "
        "python workflow/scripts/groupc/incoh_means.py "
        "--clusters {input.clusters} --arm {wildcards.arm} --out {output.npz} "
        "--shard {wildcards.i} --nshards {params.nshards} --ckpt {params.ckpt} "
        "--max_batch_tokens 3072 --chunk 128"


rule incoh_decode:
    input:
        real=expand(j(INC_DIR, "means_real_s{i}.npz"), i=range(INC_SHARDS)),
        control=expand(j(INC_DIR, "means_control_s{i}.npz"), i=range(INC_SHARDS)),
        clusters=INC_CLUSTERS,
        ckpt=ancient(QWEN_CHECKPOINT_PATH),
    output:
        labels=INC_LABELS,
    params:
        ckpt=QWEN_CHECKPOINT_PATH,
    resources:
        gpu=1,
        mem_gb=20,
    shell:
        D2L_ENV +
        "NEED_MB=20000 bash workflow/scripts/gpu_lease.sh "
        "python workflow/scripts/groupc/incoh_decode.py "
        "--real {input.real} --control {input.control} --clusters {input.clusters} "
        "--out {output.labels} --ckpt {params.ckpt}"


rule incoh_score:
    input:
        labels=INC_LABELS,
        clusters=INC_CLUSTERS,
        paper_text=ancient(j(APS_DIR, "paper_text.parquet")),
    output:
        scores=INC_SCORES,
        rows=INC_ROWS,
    params:
        nodes_eval=j(BT, "label_eval_nodes.json"),   # frozen baseline_trees artifact (see incoh_build)
        n_titles=INC_TITLES,
        n_members_sbert=INC_SBERT_N,
        seed=INC_SEED,
        use_judges=INC_JUDGES,
        sbert_model=INC_SBERT_MODEL,
    resources:
        gpu=1,
        mem_gb=30,
    script:
        "../scripts/groupc/incoh_score.py"


# The table and the figure read only the scored rows, so they are a separate CPU step: from
# the archived incoh_scores.json / incoh_rows.parquet they rebuild without SBERT, the APS
# text, or a judge call.
rule incoh_report:
    input:
        scores=INC_SCORES,
        rows=INC_ROWS,
    output:
        table=INC_TABLE,
        fig=INC_FIG,
    params:
        seed=INC_SEED,
    script:
        "../scripts/groupc/incoh_report.py"


rule groupc_incoherent:
    input:
        INC_SCORES,
        INC_TABLE,
        INC_FIG,
