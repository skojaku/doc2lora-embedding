# A small synthetic corpus the real rules run on, so the workflow can be tested
# without the licensed corpora, the checkpoints, or a GPU.
#
# The rules below call the SAME scripts the reported numbers come from --
# save_collab_scores, score_pool_field, bootstrap, tab_similarity_benchmarks,
# label_eval_metric1, label_eval_metric4 -- on vectors whose answer is known in
# advance (workflow/scripts/make_sample.py plants a method ordering). check_sample.py
# then recovers that ordering from the outputs. A pipeline that drops a method,
# unpairs the score columns, or reads a different file than it scores cannot pass.
#
# What it does not cover: gene extraction and decoding. Those need a checkpoint, and
# a synthetic corpus cannot stand in for them.
#
# RUN: snakemake sample_check -j4     # ~1 minute, CPU, no network
#      snakemake sample_judge -j1     # + the judge panel (see JUDGE_PANEL below)
from os.path import join as j

SMP = "sample"
SMP_PREP = j(DATA_DIR, "sample", "prep")
SMP_SRC = j(SMP_PREP, f"openalex-{SMP}")
SMP_FIELD_DIR = j(DATA_DIR, "fields", SMP)
SMP_EMB = j(SMP_FIELD_DIR, "embeddings")
SMP_OUT = j(DATA_DIR, "sample", "uncertainty")   # off the real tree, so a test never
SMP_POOLS = j(SMP_OUT, "pools")                 # overwrites a real score pool
SMP_LABELS = j(DATA_DIR, "sample", "labels")
SMP_WINDOWS = config.get("sample_windows", "2009,2013")
SMP_NBOOT = config.get("sample_nboot", 200)

# Every sample rule reads the corpus through bench_data, so it needs the one path
# that is not a default: where the generated openalex-<field> tables live.
SMP_ENV = (f"export FIELDS_PREP_BASE={SMP_PREP} SCORE_WINDOWS={SMP_WINDOWS} "
           f"UNCERTAINTY_OUT={SMP_OUT} LABELS_OUT={j(DATA_DIR, 'sample', 'labels')}; ")

# The judge panel to use. `local/fuzzy` answers offline by string similarity, which
# exercises the panel's plumbing -- both presentation orders, the tie rule, the
# cache -- without spending on inference. Point it at real slugs
# (e.g. "mistral-medium=mistralai/mistral-medium-3.1") for a real verdict.
SMP_JUDGES = config.get("sample_judge_panel", "local-fuzzy=local/fuzzy")


rule sample_corpus:
    input:
        script=j(SCRIPTS, "make_sample.py"),
    output:
        papers=j(SMP_SRC, "paper_table.csv"),
        authors=j(SMP_SRC, "author_paper_table.csv"),
        topics=j(SMP_FIELD_DIR, "paper_topics.parquet"),
        nodes=j(SMP_LABELS, "label_eval_nodes.json"),
        manifest=j(DATA_DIR, "sample", "manifest.json"),
    shell:
        SMP_ENV + "python {input.script} --out-prep " + SMP_PREP + " --out-data " + DATA_DIR


# The collaboration benchmark is BUILT from the generated authorship graph by the
# same script the reported numbers use, not planted: distance-2 author pairs that
# share a past collaborator, labelled by whether they later collaborate.
rule sample_collab:
    input:
        authors=j(SMP_SRC, "author_paper_table.csv"),
        script=j(SCRIPTS, "save_collab_scores.py"),
    output:
        hard=j(DATA_DIR, f"collab_scores_{SMP}_hard.parquet"),
        easy=j(DATA_DIR, f"collab_scores_{SMP}_easy.parquet"),
    shell:
        SMP_ENV + "python {input.script} --field " + SMP +
        " --negtype both --windows " + SMP_WINDOWS


rule sample_pools:
    input:
        collab=j(DATA_DIR, f"collab_scores_{SMP}_hard.parquet"),
        topics=j(SMP_FIELD_DIR, "paper_topics.parquet"),
        script=j(SCRIPTS, "score_pool_field.py"),
    output:
        expand(j(SMP_POOLS, "{task}_" + SMP + "_qwen.parquet"),
               task=["collab", "np", "topic"]),
    shell:
        SMP_ENV + "python {input.script} " + SMP + " qwen"


rule sample_bootstrap:
    input:
        pools=expand(j(SMP_POOLS, "{task}_" + SMP + "_qwen.parquet"),
                     task=["collab", "np", "topic"]),
        script=j(SCRIPTS, "bootstrap.py"),
    output:
        summary=j(SMP_OUT, "uncertainty_summary.csv"),
    shell:
        SMP_ENV + "python {input.script} " + str(SMP_NBOOT)


# The lexical metric: no model in the loop, so this reproduces exactly.
rule sample_label_metric1:
    input:
        nodes=j(SMP_LABELS, "label_eval_nodes.json"),
        script=j(SCRIPTS, "label_eval_metric1.py"),
    output:
        j(SMP_LABELS, "label_eval_metric1.json"),
    shell:
        SMP_ENV + "python {input.script}"


# The judge panel, on the same nodes. Both presentation orders are asked and
# collapsed by the real rule, so an order-dependent answer scores as a tie.
rule sample_judge:
    input:
        nodes=j(SMP_LABELS, "label_eval_nodes.json"),
        script=j(SCRIPTS, "label_eval_metric4.py"),
    output:
        j(SMP_LABELS, "label_eval_metric4.json"),
    shell:
        SMP_ENV + "set -a; source .env 2>/dev/null; set +a; "
        f"JUDGE_PANEL='{SMP_JUDGES}' python {{input.script}}"


rule sample_check:
    input:
        summary=j(SMP_OUT, "uncertainty_summary.csv"),
        metric1=j(SMP_LABELS, "label_eval_metric1.json"),
        script=j(SCRIPTS, "check_sample.py"),
    shell:
        SMP_ENV + "python {input.script} --pools " + SMP_POOLS + " --summary {input.summary}"


rule sample_check_judged:
    input:
        rules.sample_check.input,
        judge=j(SMP_LABELS, "label_eval_metric4.json"),
    shell:
        SMP_ENV + "python " + j(SCRIPTS, "check_sample.py") + " --pools " + SMP_POOLS +
        " --summary " + j(SMP_OUT, "uncertainty_summary.csv") +
        " --judge {input.judge}"
