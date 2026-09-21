# Decode-prompt paraphrase sensitivity (#73).
#
# Decoding is greedy, so each (embedding, prompt) pair yields one output, and the manuscript uses a
# different prompt per section without reporting how much an output moves under paraphrase.  This
# chain decodes every point under K=8 semantically equivalent prompts (index 0 is the verbatim
# manuscript prompt) in the three families the paper reports as text.
#
#   psens_decode -> psens_{family}.json    family in {label, describe, fusion}
#   psens_score  -> psens_scores.json + figs/prompt_sensitivity.{tex,pdf}
#
# RUN: snakemake groupc_psens -j2 --rerun-triggers mtime
from os.path import join as j

FIGS_DIR = config.get("figs_dir", "figs")
PS_DIR = j("data", "groupc", "psens")
PS_FAMILIES = config.get("psens_families", ["label", "describe", "fusion"])
PS_K = config.get("psens_k", 8)
PS_UNITS = config.get("psens_n_units", 40)
PS_SEED = config.get("psens_seed", 0)
PS_SBERT = config.get("compat_sbert_model", "sentence-transformers/all-mpnet-base-v2")
PS_MEANS = j("exps/2026-06-11-baseline-trees", "qwen_fullrank_means.npz")   # frozen artifact

PS_SCORES = j(PS_DIR, "psens_scores.json")
PS_ROWS = j(PS_DIR, "psens_rows.parquet")
PS_TABLE = j(FIGS_DIR, "prompt_sensitivity.tex")
PS_FIG = j(FIGS_DIR, "prompt_sensitivity.pdf")

D2L_SRC = config.get("doc_to_lora_src", "doc-to-lora/src")
D2L_ENV = (
    "set -a; source .env 2>/dev/null; set +a; "
    f"export DOC_TO_LORA_SRC={D2L_SRC} PYTHONPATH={D2L_SRC} "
    f"HF_HOME={config.get('hf_home', 'data/agent_assets/hf_cache')} "
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; "
)


rule psens_decode:
    input:
        sample=j("data", "groupc", "fidelity", "sample.parquet"),
        ckpt=ancient(QWEN_CHECKPOINT_PATH),
    output:
        decodes=j(PS_DIR, "psens_{family}.json"),
    params:
        means=PS_MEANS,
        ckpt=QWEN_CHECKPOINT_PATH,
        k=PS_K,
        units=PS_UNITS,
        seed=PS_SEED,
    wildcard_constraints:
        family="label|describe|fusion",
    resources:
        gpu=1,
        mem_gb=20,
    shell:
        D2L_ENV +
        "NEED_MB=20000 bash workflow/scripts/gpu_lease.sh "
        "python workflow/scripts/groupc/psens_decode.py --family {wildcards.family} "
        "--out {output.decodes} --means {params.means} --sample {input.sample} "
        "--n_units {params.units} --k {params.k} --seed {params.seed} --ckpt {params.ckpt}"


rule psens_score:
    input:
        decodes=expand(j(PS_DIR, "psens_{family}.json"), family=PS_FAMILIES),
    output:
        scores=PS_SCORES,
        rows=PS_ROWS,
        table=PS_TABLE,
        fig=PS_FIG,
    params:
        sbert_model=PS_SBERT,
    resources:
        gpu=1,
        mem_gb=16,
    script:
        "../scripts/groupc/psens_score.py"


rule groupc_psens:
    input:
        PS_SCORES,
        PS_TABLE,
        PS_FIG,
