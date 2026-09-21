# Pair-axis interpolation study (data/pair_axis) -- the data and
# the panels behind Figure 2's two-way composition result (#60 follow-up, #141).
#
# Simplifies the 3-corner fusion simplex to a single A-B axis: fuse two corner
# papers at 13 points along the interpolation, decode an abstract with each of
# the 3 channels (Doc2LoRA / In-context / ICAE), then project the decoded
# abstract's SBERT embedding back onto the A-B line to see whether the landing
# position tracks the requested mixing weight smoothly or snaps/discretizes.
#
# Corner pairs are drawn across all FIVE PACS-distance strata (L1 near ... L5
# far), 50 pairs each (250 total) -- reusing sample_far_triples.py's proven
# stratum draw (same subtopic/area/chapter/2-chapter/3-chapter rule) and keeping
# the most-separated edge of each qualifying triple as the representative pair.
#
# TWO HALVES, and they cost very differently.
#
#   kg_*   the study itself. HEAVY: 250 pairs x 13 points x {doc2lora+incontext,
#          icae} on Qwen-4B / Mistral-7B. Order of a day end-to-end even split
#          across 4 GPUs (gpu_run_kg.sh leases free GPUs dynamically). Decode is
#          batched per stratum (50 sets/job) to amortize the checkpoint load
#          across all 50 sets in that job instead of paying it 250x.
#
#   skg_*  the Figure 2 panels. CPU ONLY, seconds. Both scripts cache what they
#          plot -- pair_axis_metrics_pairaxis.json and
#          pair_axis_copyrate_pairaxis_<stratum>.json, both tracked in git --
#          and --replot draws from those caches without a GPU or SBERT. So the
#          panels rebuild from a fresh clone even though the decodes behind them
#          do not.
#
# The panel rules deliberately do NOT declare those caches as inputs, even though
# they read them. Snakemake links rules by path, so declaring KG_METRICS would
# put kg_metrics in the DAG, and kg_metrics also emits two untracked figures --
# so asking for seconds of CPU would schedule a GPU job and, in a clone without
# the gitignored corner files, the whole day-long decode chain behind it.
# ancient() does not help: it suppresses timestamp-driven reruns, not runs caused
# by a producer's other outputs being absent. The dependency is real and is
# documented here rather than declared; `kg_all` is the target that rebuilds the
# caches.
#
# RUN: snakemake simplex_kwgrid   (panels, CPU)
#      snakemake kg_all           (the full study, GPU)
from os.path import join as j

KG_DIR = "data/pair_axis"
KG_RES = j(KG_DIR, "results")
KG_STRATA = ["L1", "L2", "L3", "L4", "L5"]
KG_N = config.get("kwgrid_pairs_per_stratum", 50)
KG_IDS = [f"{n:02d}" for n in range(KG_N)]
KG_GPU = "bash workflow/scripts/gpu_run_kg.sh"
KG_D2L_SRC = config.get("doc_to_lora_src", "doc-to-lora/src")
KG_ENV = (f"export DOC_TO_LORA_SRC={KG_D2L_SRC} PYTHONPATH={KG_D2L_SRC} "
          f"DOC2LORA_CKPT={QWEN_CHECKPOINT_PATH};")

KG_CORNERS = j(KG_DIR, "corners_pair{L}_{n}.json")
KG_MANIFEST = j(KG_DIR, "pair_manifest.json")
KG_ABS = j(KG_RES, "absfollow_pair{L}_{n}_pairaxis.json")
KG_ABS_ICAE = j(KG_RES, "absfollow_pair{L}_{n}_pairaxis_icae.json")
KG_METRICS = j(KG_RES, "pair_axis_metrics_pairaxis.json")
KG_FIG_LANDING = j(KG_DIR, "figs", "pair_axis_landing.pdf")
KG_FIG_LINE = j(KG_DIR, "figs", "pair_axis_line.pdf")

# Figure 2 panels, published out of the experiment directory.
SKG_STRATA = config.get("skg_strata", ["L1", "L5"])
SKG_COPY_CACHE = j(KG_RES, "pair_axis_copyrate_pairaxis_{stratum}.json")

wildcard_constraints:
    L="|".join(KG_STRATA),


# ── The study ────────────────────────────────────────────────────────────────

# Corner-pair selection (CPU, ~2 min).
rule kg_pairs:
    input:
        script=j(SCRIPTS, "sample_far_pairs.py"),
    output:
        corners=expand(KG_CORNERS, L=KG_STRATA, n=KG_IDS),
        manifest=KG_MANIFEST,
    params:
        n=KG_N,
    shell:
        "python {input.script} --per-stratum {params.n} --factor 2 --seed 0"


# Decode (batched per stratum: 50 sets share one model load).
rule kg_decode_doc2lora:
    input:
        corners=expand(KG_CORNERS, n=KG_IDS, allow_missing=True),
        script=j(SCRIPTS, "decode_absfollow.py"),
    output:
        expand(KG_ABS, n=KG_IDS, allow_missing=True),
    params:
        sets=lambda wc: " ".join(f"pair{wc.L}_{n}" for n in KG_IDS),
    resources:
        gpu=1,
    shell:
        KG_ENV + " " + KG_GPU +
        " env SRC_KW=1 KWTAG=_pairaxis PAIR_EDGE=1 python {input.script} {params.sets}"


rule kg_decode_icae:
    input:
        corners=expand(KG_CORNERS, n=KG_IDS, allow_missing=True),
        script=j(SCRIPTS, "decode_absfollow_icae.py"),
    output:
        expand(KG_ABS_ICAE, n=KG_IDS, allow_missing=True),
    params:
        sets=lambda wc: " ".join(f"pair{wc.L}_{n}" for n in KG_IDS),
    resources:
        gpu=1,
    shell:
        KG_ENV + " " + KG_GPU +
        " env SRC_KW=1 KWTAG=_pairaxis PAIR_EDGE=1 python {input.script} {params.sets}"


# Metrics (needs 1 GPU for SBERT re-embedding of the decodes). Writes
# KG_METRICS, the cache every Figure 2 panel is drawn from.
#
# The run also drops pair_axis_landing.pdf and pair_axis_line.pdf into the
# experiment's own figs/. They are NOT declared as outputs: they are untracked,
# and a rule reruns when any declared output is missing, so declaring them made
# this GPU rule schedule itself whenever anything downstream wanted the cache --
# including figs/fig2-panels and fig2_pacs_clustering, which only ever read
# KG_METRICS and are supposed to be seconds of CPU. Before these rules existed
# nothing produced KG_METRICS, so it was a leaf and the question never arose.
rule kg_metrics:
    input:
        base=expand(KG_ABS, L=KG_STRATA, n=KG_IDS),
        icae=expand(KG_ABS_ICAE, L=KG_STRATA, n=KG_IDS),
        manifest=KG_MANIFEST,
        script=j(SCRIPTS, "pair_axis_metrics.py"),
    output:
        metrics=KG_METRICS,
    resources:
        gpu=1,
    shell:
        KG_GPU + " env KWTAG=_pairaxis python {input.script}"


rule kg_all:
    input:
        KG_METRICS,


# ── Copy rate along the edge (Fig. cluster-labels f) ────────────────────────
# Scores the same decodes kg_metrics reads, per stratum, and caches what it plots.
# Fig. cluster-labels reads this cache; the standalone per-stratum PDF the script
# also writes is an intermediate, not a reported asset, so it is not declared.
rule kg_copyrate:
    input:
        base=expand(KG_ABS, L=KG_STRATA, n=KG_IDS),
        icae=expand(KG_ABS_ICAE, L=KG_STRATA, n=KG_IDS),
        script=j(SCRIPTS, "pair_axis_copyrate.py"),
    output:
        cache=SKG_COPY_CACHE,
    resources:
        gpu=1,
    shell:
        KG_GPU + " python {input.script} --strata {wildcards.stratum}"


rule simplex_kwgrid:
    input:
        expand(SKG_COPY_CACHE, stratum=SKG_STRATA),
        KG_METRICS,


# ── Prompt sensitivity of the edge decode (§4.3 / app:prompt-sensitivity) ────
# The existing paraphrase sweep (app:prompt-sensitivity) covers a DIFFERENT prompt
# family -- "describe the combined research idea in 2-3 sentences" -- not the
# abstract instruction this edge experiment actually issues. The mixing-fidelity
# and copy-rate curves were therefore never scored per paraphrase, and §4.3 must
# not claim they are prompt-stable until they are.
#
# Prompt 0 is the manuscript's own wording, already decoded by kg_decode_* above,
# so only 1..3 are re-run. Keywords are skipped (SKIP_KW=1): only the abstract is
# scored. A reduced pair count keeps this at hours rather than days -- the claim
# under test is the SHAPE of the curve, not a tighter estimate of its height.
KG_PSENS_IDS = [str(i) for i in config.get("kwgrid_psens_prompts", [1, 2, 3])]
KG_PSENS_STRATA = ["L1", "L5"]                                  # the two strata §4.3 reports
KG_PSENS_N = config.get("kwgrid_psens_pairs", 20)
KG_PSENS_PAIRS = [f"{n:02d}" for n in range(KG_PSENS_N)]
KG_PSENS_ABS = j(KG_RES, "absfollow_pair{L}_{n}_pairaxis_p{p}.json")
KG_PSENS_ABS_ICAE = j(KG_RES, "absfollow_pair{L}_{n}_pairaxis_p{p}_icae.json")
KG_PSENS_JSON = j(KG_RES, "psens_edge.json")
KG_PSENS_TEX = j(config.get("figs_dir", "figs"), "prompt_sensitivity_edge.tex")
KG_PSENS_FIG = j(config.get("figs_dir", "figs"), "psens_edge_curves.pdf")

wildcard_constraints:
    n="[0-9]{2}",
    p="[0-9]",


rule kg_psens_decode_doc2lora:
    input:
        corners=expand(KG_CORNERS, n=KG_PSENS_PAIRS, allow_missing=True),
        script=j(SCRIPTS, "decode_absfollow.py"),
        prompts=j(SCRIPTS, "psens_prompts.py"),
    output:
        expand(KG_PSENS_ABS, n=KG_PSENS_PAIRS, allow_missing=True),
    params:
        sets=lambda wc: " ".join(f"pair{wc.L}_{n}" for n in KG_PSENS_PAIRS),
    resources:
        gpu=1,
    shell:
        KG_ENV + " " + KG_GPU +
        " env SRC_KW=1 KWTAG=_pairaxis_p{wildcards.p} PAIR_EDGE=1 SKIP_KW=1"
        " ABS_PROMPT_ID={wildcards.p} python {input.script} {params.sets}"


rule kg_psens_decode_icae:
    input:
        corners=expand(KG_CORNERS, n=KG_PSENS_PAIRS, allow_missing=True),
        script=j(SCRIPTS, "decode_absfollow_icae.py"),
        prompts=j(SCRIPTS, "psens_prompts.py"),
    output:
        expand(KG_PSENS_ABS_ICAE, n=KG_PSENS_PAIRS, allow_missing=True),
    params:
        sets=lambda wc: " ".join(f"pair{wc.L}_{n}" for n in KG_PSENS_PAIRS),
    resources:
        gpu=1,
    shell:
        KG_ENV + " " + KG_GPU +
        " env SRC_KW=1 KWTAG=_pairaxis_p{wildcards.p} PAIR_EDGE=1 SKIP_KW=1"
        " ABS_PROMPT_ID={wildcards.p} python {input.script} {params.sets}"


rule kg_psens_score:
    input:
        base=expand(KG_PSENS_ABS, L=KG_PSENS_STRATA, n=KG_PSENS_PAIRS, p=KG_PSENS_IDS),
        icae=expand(KG_PSENS_ABS_ICAE, L=KG_PSENS_STRATA, n=KG_PSENS_PAIRS, p=KG_PSENS_IDS),
        p0=expand(KG_ABS, L=KG_PSENS_STRATA, n=KG_PSENS_PAIRS),
        p0_icae=expand(KG_ABS_ICAE, L=KG_PSENS_STRATA, n=KG_PSENS_PAIRS),
        script=j(SCRIPTS, "psens_edge_score.py"),
    output:
        js=KG_PSENS_JSON,
        tex=KG_PSENS_TEX,
        fig=KG_PSENS_FIG,
    params:
        n=KG_PSENS_N,
        ids=" ".join(KG_PSENS_IDS),
    resources:
        gpu=1,
    shell:
        KG_GPU + " python {input.script} --pairs {params.n} --prompts 0 {params.ids}"
        " --json {output.js} --tex {output.tex} --fig {output.fig}"


rule kg_psens_all:
    input:
        KG_PSENS_JSON,
        KG_PSENS_TEX,
        KG_PSENS_FIG,
