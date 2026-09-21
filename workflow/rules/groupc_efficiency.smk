# Efficiency and storage cost (#71) -- the exchange rate the paper never shows.
#
# The manuscript claims corpus-scale embedding and concedes the dimensionality in Limitation 3, but
# never reports the cost of a full base-LM forward pass per document against a 110M-parameter encoder
# that wins 10 of 14 benchmarks.  This chain measures it:
#
#   eff_throughput -> throughput_{method}.json   docs/s, peak VRAM, params (one process per method)
#   eff_index      -> eff_index.json             bytes/paper, index size at 2.2M, PQ recall@10
#   eff_report     -> eff_summary.csv + figs/efficiency.tex + figs/efficiency_index.tex
#
# RUN: snakemake groupc_efficiency -j1 --rerun-triggers mtime
from os.path import join as j

FIGS_DIR = config.get("figs_dir", "figs")
EFF_DIR = j("data", "groupc", "efficiency")
EFF_METHODS = config.get("eff_methods", ["sbert", "specter2", "icae", "d2l-qwen", "d2l-mistral"])
EFF_NDOCS = config.get("eff_n_docs", 256)
EFF_CORPUS = config.get("eff_corpus_size", 2200000)
EFF_REPS = config.get("eff_reps", {
    "Doc2LoRA rank-pooled (Qwen)": ["qwen_norm_lora_emb.npz", "embeddings"],
    "ICAE mean-slot": ["icae_emb.npz", "vecs"],
    "SBERT all-mpnet": ["sbert_allmpnet.npz", "vecs"],
    "SPECTER2": ["baseline_specter2.npz", "vecs"],
})
EFF_PQ_BYTES = config.get("eff_pq_bytes", [1152, 288, 96])
EFF_NTRAIN = config.get("eff_n_train", 50000)
EFF_NBASE = config.get("eff_n_base", 50000)
EFF_NQUERY = config.get("eff_n_query", 2000)

EFF_INDEX = j(EFF_DIR, "eff_index.json")
EFF_CSV = j(EFF_DIR, "eff_summary.csv")
EFF_TABLE = j(FIGS_DIR, "efficiency.tex")
EFF_INDEX_TABLE = j(FIGS_DIR, "efficiency_index.tex")

D2L_SRC = config.get("doc_to_lora_src", "doc-to-lora/src")
D2L_ENV = (
    "set -a; source .env 2>/dev/null; set +a; "
    f"export DOC_TO_LORA_SRC={D2L_SRC} PYTHONPATH={D2L_SRC} "
    f"HF_HOME={config.get('hf_home', 'data/agent_assets/hf_cache')} "
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; "
)


rule eff_throughput:
    input:
        sample=j("data", "groupc", "fidelity", "sample.parquet"),
    output:
        json=j(EFF_DIR, "throughput_{method}.json"),
    params:
        n_docs=EFF_NDOCS,
    wildcard_constraints:
        method="|".join(EFF_METHODS),
    resources:
        gpu=1,
        mem_gb=30,
    shell:
        D2L_ENV +
        "NEED_MB=26000 bash workflow/scripts/gpu_lease.sh "
        "python workflow/scripts/groupc/eff_throughput.py --method {wildcards.method} "
        "--sample {input.sample} --out {output.json} --n_docs {params.n_docs}"


rule eff_index:
    input:
        genes=ancient(j(EMB_DIR, "qwen_norm_lora_emb.npz")),
    output:
        json=EFF_INDEX,
    params:
        reps=EFF_REPS,
        emb_dir=EMB_DIR,
        n_train=EFF_NTRAIN,
        n_base=EFF_NBASE,
        n_query=EFF_NQUERY,
        corpus_size=EFF_CORPUS,
        pq_bytes=EFF_PQ_BYTES,
        seed=0,
    resources:
        mem_gb=120,
    script:
        "../scripts/groupc/eff_index.py"


rule eff_report:
    input:
        throughput=expand(j(EFF_DIR, "throughput_{method}.json"), method=EFF_METHODS),
        index=EFF_INDEX,
    output:
        csv=EFF_CSV,
        table=EFF_TABLE,
        index_table=EFF_INDEX_TABLE,
    resources:
        mem_gb=8,
    script:
        "../scripts/groupc/eff_report.py"


rule groupc_efficiency:
    input:
        EFF_CSV,
        EFF_TABLE,
        EFF_INDEX_TABLE,
