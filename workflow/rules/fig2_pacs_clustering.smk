# Figure 2 (<figs_dir>/pacs-clustering.pdf) and Tab. 1 mixing-decode
# (<figs_dir>/mixing_decode.tex): cluster labels and mixtures.
#
# Both are drawn from cached numbers by scripts under workflow/:
#
#   Fig. 2 (workflow/plot/fig_pacs_clustering.py), 2 x 3
#   (a) fuzzy overlap per node     <- baseline_trees.smk: bt_label_eval_metric1
#   (b) judge round robin          <- baseline_trees.smk: bt_label_eval_metric4 (OpenRouter judges)
#   (c) PACS breadth vs depth      <- baseline_trees.smk: bt_length_dial (csv)
#                                     + bt_decode_fullrank (decoded labels)
#   (d) Wikipedia abstraction walk <- abstraction_walk.smk: aw_doc2lora
#   (e) actual vs ideal mixing     <- pair_axis_metrics.py cache  (simplex_kwgrid.smk)
#   (f) verbatim copy rate         <- pair_axis_copyrate.py cache (simplex_kwgrid.smk)
#       + T2L, same grid           <- fig2_t2l_edge_metrics (SBERT on CPU over the T2L
#                                     edge decodes of t2l.smk:t2l_fusion)
#   Tab. mixing-decode (workflow/scripts/fig2_mixing_table.py)
#       one pair decoded at 3/12, 5/12, 9/12 of the way to paper B: the decoded abstracts' openings from the
#       archived pairCSML_00 decodes, t from fig2_colorband_metrics (SBERT on CPU)
#
# CPU only. Every input is a small cache that ships in the `results` artifact tier
# (scripts/fetch_artifacts.py results), so the assets rebuild on a CPU without the
# GPU decodes or judge calls behind those caches. Regenerating a cache itself is
# the job of the rule named next to it.
#
# RUN: snakemake fig2
from os.path import join as j

FIG2_BT = "data/labels"
FIG2_SKG = "data/pair_axis"
FIG2_PAIR_SET = config.get("fig2_pair_set", "pairCSML_00")   # the pair shown in the table
FIG2_STRATUM = config.get("fig2_stratum", "L1")               # the stratum plotted in (e), (f)
FIG2_T2L_SPACE = config.get("fig2_t2l_space", "e1")           # T2L embedding position: e0 gte, e1 TaskEncoder, e2 LoRA factors
FIG2_T2L_MIDPOINTS = j("data/t2l", "results", f"midpoints_t2l_{FIG2_T2L_SPACE}.json")
FIG2_T2L_CACHE = j(FIG2_SKG, "results", f"pair_axis_t2l_{FIG2_T2L_SPACE}_pairaxis.json")

FIG2_DECODES = j(FIG2_SKG, "results", f"absfollow_{FIG2_PAIR_SET}_pairaxis.json")
FIG2_DECODES_ICAE = j(FIG2_SKG, "results", f"absfollow_{FIG2_PAIR_SET}_pairaxis_icae.json")
FIG2_BAND_METRICS = j(FIG2_SKG, "results", f"colorband_{FIG2_PAIR_SET}_pairaxis.json")
FIG2_PDF = j(config.get("figs_dir", "figs"), "pacs-clustering.pdf")
FIG2_TABLE = j(config.get("figs_dir", "figs"), "mixing_decode.tex")


# Code inputs of the rules that decode, extract, or call a model are ancient(): a fresh clone
# checks every script out newer than the archived outputs (the `results` tier), and an mtime
# trigger on the script would otherwise re-run a GPU job whose result already ships. After
# editing such a script, rerun its rule with -R <rule>. Table and figure rules keep plain
# script inputs.

# SBERT projection + copy rate per decoded cell, and the corner papers'
# titles/DOIs. all-mpnet-base-v2 on CPU. The corner spec carries the paper ids
# and DOIs; the two optional params are fallbacks for a spec without them, read
# only when the files exist on this machine (manifest = ids, APS text = DOIs).
# ── The one curated pair Tab. mixing-decode shows ───────────────────────────
# The 250 pairs of simplex_kwgrid are a stratified random draw; this is a single
# hand-picked pair (a statistical-mechanics paper and a quantum-error-correction
# paper) whose corners ship with the workflow, decoded by the same two scripts on
# the same 13-point grid. It sits here rather than in simplex_kwgrid.smk because
# its set name is outside that file's L1..L5 wildcard pattern.
rule fig2_decode_curated:
    input:
        corners = ancient(j(SCRIPTS, f"corners_{FIG2_PAIR_SET}.json")),
        script = ancient(j(SCRIPTS, "decode_absfollow.py")),
        prompts = ancient(j(SCRIPTS, "psens_prompts.py")),
    output:
        FIG2_DECODES,
    resources:
        gpu = 1,
    shell:
        KG_ENV + " " + KG_GPU +
        " env SRC_KW=1 KWTAG=_pairaxis PAIR_EDGE=1 python {input.script} " + FIG2_PAIR_SET


rule fig2_decode_curated_icae:
    input:
        corners = ancient(j(SCRIPTS, f"corners_{FIG2_PAIR_SET}.json")),
        script = ancient(j(SCRIPTS, "decode_absfollow_icae.py")),
        prompts = ancient(j(SCRIPTS, "psens_prompts.py")),
    output:
        FIG2_DECODES_ICAE,
    resources:
        gpu = 1,
    shell:
        KG_ENV + " " + KG_GPU +
        " env SRC_KW=1 KWTAG=_pairaxis PAIR_EDGE=1 python {input.script} " + FIG2_PAIR_SET


# Inputs and output are tracked in git or ship in the `results` tier, and either way they carry
# arbitrary mtimes -- so without ancient() this CPU+SBERT job re-runs on a clone
# purely to rewrite a file that is already correct.
rule fig2_colorband_metrics:
    input:
        corners = ancient(j(SCRIPTS, f"corners_{FIG2_PAIR_SET}.json")),
        decodes = FIG2_DECODES,
        icae = FIG2_DECODES_ICAE,
    output:
        metrics = FIG2_BAND_METRICS,
    params:
        manifest = j(FIG2_SKG, "pair_manifest.json"),
        paper_text = BT_PAPER_TEXT,
        device = "cpu",
    script:
        "../scripts/fig2_colorband_metrics.py"


# T2L along the edge: t2l_fusion.py decoded 100 corner pairs on the 13-point alpha
# grid of the other methods (GPU); this scores those decodes the same
# way, so (e), (f) carry a T2L curve rather than three markers. The
# cache ships in the `results` tier, so a clone without those decodes still draws (e), (f).
rule fig2_t2l_edge_metrics:
    input:
        midpoints = ancient(FIG2_T2L_MIDPOINTS),
    output:
        cache = FIG2_T2L_CACHE,
    params:
        device = "cpu",
    script:
        "../scripts/fig2_t2l_edge_metrics.py"


rule fig2_mixing_table:
    input:
        band_json = FIG2_BAND_METRICS,
        decodes = FIG2_DECODES,
        icae = FIG2_DECODES_ICAE,
    output:
        tex = FIG2_TABLE,
    script:
        "../scripts/fig2_mixing_table.py"


# The judge cache (panel b) is deliberately a PARAM, not an input: as an input
# it would pull bt_label_eval_metric4 -- the OpenRouter judge panel -- into the
# `paper_assets` DAG, which baseline_trees keeps out on purpose (it sits behind the
# separate `label_eval` target). The file ships in the `results` tier; after a new judge
# run, `snakemake fig2 --forcerun fig2_pacs_clustering` re-plots it.
rule fig2_pacs_clustering:
    input:
        fuzzy_json = BT_EVAL_M1,
        radius_csv = BT_RADIUS_CSV,
        labels_json = BT_FULLRANK,
        walk_json = AW_DOC2LORA,
        metrics_json = j(FIG2_SKG, "results", "pair_axis_metrics_pairaxis.json"),
        copyrate_json = j(FIG2_SKG, "results", f"pair_axis_copyrate_pairaxis_{FIG2_STRATUM}.json"),
        t2l_json = FIG2_T2L_CACHE,
        palette = "workflow/plot/_fig2_palette.py",
    output:
        pdf = FIG2_PDF,
    params:
        judge_json = BT_EVAL_M4,
        stratum = FIG2_STRATUM,
    script:
        "../plot/fig_pacs_clustering.py"


rule fig2:
    input:
        rules.fig2_pacs_clustering.output,
        rules.fig2_mixing_table.output,
