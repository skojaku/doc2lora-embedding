# Scientific-text embedding BASELINES (SPECTER2 / INSTRUCTOR / text2vec / SBERT)
# embedded per FIELD, to be scored head-to-head against doc2lora idea-genes on the
# mobility-compatibility test and retrieval.
#
# Input : data/fields/<field>/paper_text.parquet  (paper_id, title, abstract, text)
# Output: data/fields/<field>/embeddings/baseline_<method>.npz  {paper_ids, vecs}
#
# Encoders live in workflow/scripts/text_encoders.py (shared registry); the embedder
# is workflow/scripts/embed_text_baselines_field.py (dual-mode).

from os.path import join as j

FIELDS_DIR = j(DATA_DIR, "fields")

BASELINE_METHODS = config.get("baseline_methods", ["specter2", "instructor", "text2vec", "sbert", "embeddinggemma", "gte"])
BASELINE_FIELDS = config.get("baseline_fields", ["economics"])

BASELINE_EMB = j(FIELDS_DIR, "{field}", "embeddings", "baseline_{method}.npz")
EMGEMMA_VENV = ".venv-emgemma"


# EmbeddingGemma (Gemma3 arch) needs transformers>=4.56; the main env is pinned to 4.51.3 for
# doc2lora, so it runs in an isolated --system-site-packages venv (see script header). Reproducible.
rule setup_emgemma_venv:
    output: touch(j(EMGEMMA_VENV, ".ready")),
    shell: "bash workflow/scripts/setup_emgemma_venv.sh"


def _baseline_inputs(w):
    ins = {"paper_text": j(FIELDS_DIR, w.field, "paper_text.parquet")}
    if w.method == "embeddinggemma":          # only this method needs the newer-transformers venv
        ins["venv"] = j(EMGEMMA_VENV, ".ready")
    return ins


# Dual interpreter: EmbeddingGemma -> .venv-emgemma python; all other baselines -> main pinned env.
# Shell (not script:) so the interpreter can switch per method; the embedder supports an argparse CLI.
rule embed_text_baseline:
    input:
        unpack(_baseline_inputs),
    output:
        embeddings=BASELINE_EMB,
    # Per-rule constraints: s2and.smk installs a GLOBAL `method` constraint
    # (sbert|instructor|embeddinggemma) that would otherwise stop this rule from
    # producing baseline_specter2.npz on a fresh checkout.
    wildcard_constraints:
        field="|".join(BASELINE_FIELDS),
        method="|".join(BASELINE_METHODS),
    params:
        py=lambda w: j(EMGEMMA_VENV, "bin", "python") if w.method == "embeddinggemma" else "python",
        batch_size=lambda w: config.get("emgemma_batch_size", 128) if w.method == "embeddinggemma"
                             else config.get("baseline_batch_size", 64),
        chunk_size=config.get("baseline_chunk_size", 20000),
    resources:
        gpu=1,
    shell:
        "set -a; source .env 2>/dev/null; set +a; "
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        "{params.py} workflow/scripts/embed_text_baselines_field.py "
        "--paper-text {input.paper_text} --method {wildcards.method} "
        "--output {output.embeddings} --gpu --batch-size {params.batch_size} --chunk-size {params.chunk_size}"


rule baselines_all:
    input:
        expand(BASELINE_EMB, field=BASELINE_FIELDS, method=BASELINE_METHODS),


# APS text baselines. The field rule above is wildcard-scoped to data/fields/...,
# so the APS copies of the same baselines need their own rule: same script, same
# output convention, keyed by aps_paper_id. `unc_pool_field` reads all three.
APS_BASELINE_EMB = j(DATA_DIR, "aps", "embeddings", "baseline_{method}.npz")
APS_BASELINE_METHODS = config.get("aps_baseline_methods", ["specter2", "instructor", "gte"])


rule embed_aps_text_baseline:
    input:
        paper_text=j(DATA_DIR, "aps", "paper_text.parquet"),
    output:
        embeddings=APS_BASELINE_EMB,
    params:
        batch_size=config.get("baseline_batch_size", 64),
        chunk_size=config.get("baseline_chunk_size", 20000),
    wildcard_constraints:
        method="|".join(APS_BASELINE_METHODS),
    resources:
        gpu=1,
    shell:
        "set -a; source .env 2>/dev/null; set +a; "
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        "python workflow/scripts/embed_text_baselines_field.py "
        "--paper-text {input.paper_text} --method {wildcards.method} "
        "--output {output.embeddings} --gpu --batch-size {params.batch_size} "
        "--chunk-size {params.chunk_size}"


rule baselines_aps:
    input:
        expand(APS_BASELINE_EMB, method=APS_BASELINE_METHODS),
