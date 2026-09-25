"""Reproduction workflow for "Doc2LoRA Provides Decodable Representations of
Scientific Ideas".

Every rule here produces a number, a table, or a figure that the paper reports.
A chain whose result did not reach the paper is not in this repository, and a
rule that no reported result depends on is not in this workflow.

Entry points
------------
    snakemake -n paper_assets     # dry run: prints the whole DAG, runs nothing
    snakemake paper_assets -j4    # rebuild every reported asset a rule owns
    snakemake sample_check -j4    # the same pipeline on the bundled sample corpus

Step 0 is mandatory: copy workflow/config.template.yaml to workflow/config.yaml
and set the paths for your machine. See README.md and REPRODUCE.md.
"""

import os
from os.path import join as j

configfile: "workflow/config.yaml"

include: "workflow/workflow_utils.smk"

# ── Checkpoints ──────────────────────────────────────────────────────────
# Gemma resolves from the config value first, then $DOC2LORA_CKPT.
CHECKPOINT_PATH = config.get("checkpoint_path") or os.environ.get("DOC2LORA_CKPT")
MISTRAL_CHECKPOINT_PATH = config["mistral_checkpoint_path"]
QWEN_CHECKPOINT_PATH = config["qwen_checkpoint_path"]

# ── Paths ────────────────────────────────────────────────────────────────
DATA_DIR = config["data_dir"]
APS_DIR = j(DATA_DIR, "aps")
EMB_DIR = j(APS_DIR, "embeddings")
POOLING_CSV = j(APS_DIR, "pooling_spearman.csv")

# Every script the rules call lives in one directory; every file a rule writes
# lives under data/. Rule files read both from the including scope.
SCRIPTS = "workflow/scripts"
PLOT = "workflow/plot"

# Where the generated tables and figures land. Nothing here is tracked: every
# file under it is the output of a rule in this workflow.
FIGS_DIR = config.get("figs_dir", "results/figs")

# ── Sub-workflows (closure of the manuscript's assets) ───────────────────
# Shared backbone: corpora -> genes -> transforms -> per-task scores
include: "workflow/rules/pacs_groups.smk"       # PACS concept hierarchy (node set for labelling)
include: "workflow/rules/fields.smk"            # economics / psychology corpora + genes
include: "workflow/rules/baselines.smk"         # SPECTER2 / INSTRUCTOR / EmbeddingGemma / GTE
include: "workflow/rules/collab_scores.smk"     # co-authorship benchmark pairs
include: "workflow/rules/idea_compatibility.smk"  # full-corpus SBERT vectors over APS abstracts
include: "workflow/rules/kron_adapter.smk"      # per-field invertible citation adapter
include: "workflow/rules/general_adapter.smk"   # one general OpenAlex adapter (genkron)
include: "workflow/rules/s2and.smk"             # author-name disambiguation benchmark
# Manuscript assets
include: "workflow/rules/uncertainty.smk"       # Tab. similarity + Tab. encoder-matrix
include: "workflow/rules/baseline_trees.smk"    # Tab. hierarchy-labels + Tab. label-eval
include: "workflow/rules/abstraction_walk.smk"  # Wikipedia radius walk (Fig. cluster-labels panel d)
include: "workflow/rules/fid_sample.smk"        # the abstract sample the psens decodes share
include: "workflow/rules/groupc_psens.smk"      # Tab. prompt-sensitivity (App.)
include: "workflow/rules/groupc_incoherent.smk" # Tab. incoherent-control (App.)
# Chains whose numbers are TYPED into the manuscript rather than \input-ed, so they
# sit behind their own targets and not in `paper_assets` (see REPRODUCE.md):
include: "workflow/rules/bench.smk"             # the sliced benchmark subsets
include: "workflow/rules/groupc_bench.smk"      # temporal-hardening claims (App. datasets, #72)
include: "workflow/rules/icae.smk"              # ICAE slots + its citation transform
include: "workflow/rules/groupc_s2and.smk"      # App. symmetric-adapter: the S2AND half, the
                                                # raw-score table, the gain figure, head-to-head
include: "workflow/rules/t2l.smk"               # Text-to-LoRA hypernetwork-adapter baseline
include: "workflow/rules/actpatch.smk"          # the base model's own hidden states as a decoder
include: "workflow/rules/recipe_fusion.smk"     # the recipe blend quoted in App. recipe-fusion

# The pair-axis interpolation study behind Sec. results' two-way composition and
# App. prompt-sensitivity's edge sweep. Include it ONCE and before fig2: Snakemake
# does not deduplicate includes, and a second one aborts the workflow with
# "The name kg_pairs is already used". t2l.smk above reads its corner specs, which
# are regenerable artifacts, so without this include `snakemake t2l_all` resolves
# in a warm tree and dead-ends on a fresh clone.
include: "workflow/rules/simplex_kwgrid.smk"    # Fig. cluster-labels panels (e), (f) + Fig. psens-edge
include: "workflow/rules/fig2_pacs_clustering.smk"  # Fig. cluster-labels + Tab. mixing-decode

# A small synthetic corpus the rules above can be tested on, without the licensed
# corpora, the checkpoints, or a GPU. `snakemake sample_check` is the whole test.
include: "workflow/rules/sample.smk"

# ── Manuscript targets ───────────────────────────────────────────────────
#
# `paper_assets` builds exactly what paper/iclr2026 \input's or \includegraphics's
# and that a rule owns. Adding a rule here is a claim that the manuscript reads
# its output. Assets with no rule (Fig. method, Fig. cluster-labels) are listed
# in REPRODUCE.md with the command that makes them.

rule paper_assets:
    input:
        # Tab. similarity        -> figs/similarity_benchmarks.tex
        # Tab. encoder-matrix    -> figs/encoder_matrix.tex
        rules.uncertainty_all.input,
        # Tab. hierarchy-labels  -> paper/iclr2026/hierarchy_rows.tex
        # plus the cluster-label report behind Tab. label-eval. The LLM judge
        # panel (metric 4) is NOT pulled in here; run `label_eval` for it.
        rules.baseline_trees.input,
        # Tab. prompt-sensitivity -> figs/prompt_sensitivity.tex   (App., #73)
        rules.groupc_psens.input,
        # Tab. incoherent-control -> figs/incoherent_control.tex   (App., #101)
        rules.groupc_incoherent.input,
        # Fig. cluster-labels     -> figs/pacs-clustering.pdf
        # Tab. mixing-decode      -> figs/mixing_decode.tex           (Sec. results)
        rules.fig2.input,
        # Fig. psens-edge -> figs/psens_edge_curves.pdf               (App.)
        # The same rule also writes figs/prompt_sensitivity_edge.tex, whose numbers
        # the appendix states in prose rather than \input-ing.
        rules.kg_psens_all.input,
        # Tab. symmetric-raw       -> <figs_dir>/symmetric_raw_scores.tex
        # Tab. symmetric-summary   -> <figs_dir>/symmetric_adapter_summary.tex
        # Fig. symmetric-gain      -> <figs_dir>/symmetric_adapter_gain.pdf   (App.)
        # Tab. collab-per-window   -> <figs_dir>/collab_per_window.tex        (App. tables)
        rules.groupc_s2and.input,
        # Spearman rho quoted in Sec. methods (mean-over-rank vs full tensor).
        # Emits the CSV only; the number is transcribed into the text by hand.
        POOLING_CSV,


# ── Convenience targets (each sub-workflow on its own) ───────────────────

rule all:
    input:
        j(APS_DIR, "paper_text.parquet"),

rule fields:
    input:
        rules.fields_all.input,

rule baselines:
    input:
        rules.baselines_all.input,

rule kron:
    input:
        rules.kron_all.input,

rule general:
    input:
        rules.ga_all.input,

rule leakage:
    input:
        rules.ga_leakage.input,

rule s2and:
    input:
        rules.s2and_all.input,

rule uncertainty:
    input:
        rules.uncertainty_all.input,

rule temporal_hardening:
    input:
        rules.groupc_bench.input,

rule figure2:
    input:
        rules.fig2.input,

rule t2l:
    input:
        rules.t2l_all.input,

rule recipe_fusion:
    input:
        rules.recipe_fusion_all.input,

rule pair_axis:
    input:
        rules.kg_all.input,

rule all_embeddings:
    input:
        j(EMB_DIR, "gemma_norm_lora_emb.npz"),
        j(EMB_DIR, "mistral_norm_lora_emb.npz"),
        j(EMB_DIR, "qwen_norm_lora_emb.npz"),


# ── Data preparation ────────────────────────────────────────────────────

rule prepare_aps_text:
    input:
        aps_papers=config["aps_paper_table"],
        openalex_papers=config["openalex_paper_table"],
        openalex_abstracts=config["openalex_abstracts"],
    output:
        paper_text=j(APS_DIR, "paper_text.parquet"),
        report=j(APS_DIR, "matching_report.md"),
    script:
        "workflow/scripts/prepare_aps_text.py"


# The same table keyed on `paper_id` instead of `aps_paper_id`. The label chains
# (prep_nodes, vec2text) expect that spelling; upstream the file existed on disk
# with no producing rule. Verified to be a pure column rename: identical row
# count, identical ids.

rule aps_text_pid:
    input:
        paper_text=j(APS_DIR, "paper_text.parquet"),
    output:
        paper_text_pid=j(APS_DIR, "paper_text_pid.parquet"),
    run:
        import pandas as pd
        df = pd.read_parquet(input.paper_text)
        df = df.rename(columns={"aps_paper_id": "paper_id"})
        df["paper_id"] = df["paper_id"].astype("int64")
        df.to_parquet(output.paper_text_pid, index=False)


# ── Mean-pooled idea genes (one rule per encoder) ────────────────────────

rule embed_aps_papers:
    input:
        paper_text=j(APS_DIR, "paper_text.parquet"),
    output:
        embeddings=j(EMB_DIR, "gemma_norm_lora_emb.npz"),
    params:
        checkpoint_path=CHECKPOINT_PATH,
        shard_dir=j(EMB_DIR, "shards_norm_lora_emb"),
        shard_size=config["shard_size"],
        gpu_ids=config["gpu_ids"],
    resources:
        gpu=1,
    script:
        "workflow/scripts/embed_aps_papers.py"


rule embed_aps_papers_mistral:
    input:
        paper_text=j(APS_DIR, "paper_text.parquet"),
    output:
        embeddings=j(EMB_DIR, "mistral_norm_lora_emb.npz"),
    params:
        checkpoint_path=MISTRAL_CHECKPOINT_PATH,
        shard_dir=j(EMB_DIR, "mistral_shards_norm_lora_emb"),
        shard_size=config["shard_size"],
        gpu_ids=config["gpu_ids"],
    resources:
        gpu=1,
    script:
        "workflow/scripts/embed_aps_papers_mistral.py"


rule embed_aps_papers_qwen:
    input:
        paper_text=j(APS_DIR, "paper_text.parquet"),
    output:
        embeddings=j(EMB_DIR, "qwen_norm_lora_emb.npz"),
    params:
        checkpoint_path=QWEN_CHECKPOINT_PATH,
        shard_dir=j(EMB_DIR, "qwen_shards_norm_lora_emb"),
        shard_size=config["shard_size"],
        gpu_ids=config["gpu_ids"],
    resources:
        gpu=1,
    script:
        "workflow/scripts/embed_aps_papers_qwen.py"


# ── Pooling validation (mean-over-rank vs full tensor, App.) ─────────────

rule pooling_validation:
    input:
        paper_text=j(APS_DIR, "paper_text.parquet"),
    output:
        csv=POOLING_CSV,
    params:
        n_docs=config.get("pooling_n_docs", 500),
        seed=config.get("pooling_seed", 42),
    resources:
        gpu=1,
    script:
        "workflow/scripts/pooling_validation.py"
