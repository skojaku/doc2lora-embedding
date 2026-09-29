# S2AND author-name-disambiguation with idea genes (data/s2and).
#
# Per dataset {ds} (AllenAI S2AND public S3) and gene encoder {enc} in {gemma,qwen,mistral}:
#   s2and_download  : aws s3 sync the raw {signatures,papers,clusters}.json + specter.pickle
#   s2and_prep      : -> sig_table.parquet + paper_text.parquet + specter.npz
#   s2and_genes     : doc2lora gene extraction (lib mode=embed, capped tokens) -> genes_{enc}.npz   [GPU]
#   s2and_text      : sbert / instructor baselines -> {method}.npz                                  [GPU]
#   s2and_eval      : embedding-isolation AND (gene/gene_kron/specter/sbert/instructor) -> results_{ds}_{enc}.csv
#
# Scripts are argparse CLIs (root-relative paths); GPU rules go through gpu_run.sh which leases a free GPU,
# so `snakemake s2and --rerun-triggers mtime -j2` runs two GPU jobs on distinct free GPUs without colliding.
import os
from os.path import join as j

S2DIR = "data/s2and"
S2_BUCKET = "s3://ai2-s2-research-public/s2and-release"
# clusters-based datasets only (medline ships pairwise train/test_pairs, NO clusters.json -> excluded).
# the two giants (inspire ~750k, aminer ~900k papers) are excluded for runtime; add either via config.
S2AND_DATASETS = config.get("s2and_datasets",
                            ["zbmath", "qian", "arnetminer", "pubmed", "kisti"])
S2AND_ENC = config.get("s2and_encoders", ["gemma", "qwen", "mistral"])
S2AND_TEXT = ["sbert", "instructor", "embeddinggemma", "gte"]   # embeddinggemma embeds in .venv-emgemma
EMGEMMA_PY = ".venv-emgemma/bin/python"                  # transformers>=4.56 (main env is pinned 4.51.3)

S2_HF = os.path.abspath("data/agent_assets/hf_cache")
S2_ENV = (f"set -a; source .env 2>/dev/null || true; set +a; "
          f"export HF_HOME={S2_HF} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;")
GPU = f"bash {SCRIPTS}/gpu_run.sh"

RAW = j(S2DIR, "data", "{ds}")
PROC = j(S2DIR, "proc", "{ds}")

wildcard_constraints:
    ds="|".join(S2AND_DATASETS),
    enc="|".join(S2AND_ENC),
    method="|".join(S2AND_TEXT),


rule s2and_download:
    output:
        sig=j(RAW, "{ds}_signatures.json"), pap=j(RAW, "{ds}_papers.json"),
        clu=j(RAW, "{ds}_clusters.json"), spec=j(RAW, "{ds}_specter.pickle"),
    shell:
        f"aws s3 sync --no-sign-request --quiet {S2_BUCKET}/{{wildcards.ds}}/ {S2DIR}/data/{{wildcards.ds}}/"

rule s2and_prep:
    input:
        sig=j(RAW, "{ds}_signatures.json"), pap=j(RAW, "{ds}_papers.json"),
        clu=j(RAW, "{ds}_clusters.json"), spec=j(RAW, "{ds}_specter.pickle"),
    output:
        sig=j(PROC, "sig_table.parquet"), txt=j(PROC, "paper_text.parquet"), spec=j(PROC, "specter.npz"),
    resources: mem_gb=20,
    shell:
        f"python {SCRIPTS}/prep.py {{wildcards.ds}}"

rule s2and_genes:
    input: txt=j(PROC, "paper_text.parquet"),
    output: npz=j(PROC, "genes_{enc}.npz"),
    params: batch=lambda w: "4" if w.enc == "mistral" else "8",
    resources: gpu=1, mem_gb=40,
    shell:
        f"{S2_ENV} MAX_TOKENS=512 BATCH={{params.batch}} {GPU} python {SCRIPTS}/extract_genes.py {{wildcards.ds}} {{wildcards.enc}}"

def _s2and_text_inputs(w):
    ins = {"txt": j(PROC, "paper_text.parquet").format(ds=w.ds)}
    if w.method == "embeddinggemma":
        ins["venv"] = ".venv-emgemma/.ready"
    return ins


rule s2and_text:
    input: unpack(_s2and_text_inputs),
    output: npz=j(PROC, "{method}.npz"),
    # EmbeddingGemma -> .venv-emgemma python (transformers>=4.56); other text baselines -> main env
    params: py=lambda w: EMGEMMA_PY if w.method == "embeddinggemma" else "python",
    resources: gpu=1, mem_gb=20,
    shell:
        f"{S2_ENV} {GPU} {{params.py}} {SCRIPTS}/text_baselines.py {{wildcards.ds}} {{wildcards.method}}"

rule s2and_eval:
    input:
        gene=j(PROC, "genes_{enc}.npz"), spec=j(PROC, "specter.npz"),
        sbert=j(PROC, "sbert.npz"), instr=j(PROC, "instructor.npz"),
        emgemma=j(PROC, "embeddinggemma.npz"),
        gte=j(PROC, "gte.npz"),
    output: csv=j(S2DIR, "results_{ds}_{enc}.csv"),
    resources: gpu=1, mem_gb=40,
    shell:
        f"{S2_ENV} KRON_STEPS=2000 KRON_REG=10 {GPU} python {SCRIPTS}/and_eval.py {{wildcards.ds}} {{wildcards.enc}}"

def _s2and_targets():
    out = []
    for ds in S2AND_DATASETS:
        for enc in S2AND_ENC:
            out.append(j(S2DIR, f"results_{ds}_{enc}.csv"))
    return out

rule s2and_all:
    input:
        _s2and_targets(),
