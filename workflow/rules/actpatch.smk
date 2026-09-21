# ActPatch: the base model's own hidden states as a decoder baseline (App. actpatch).
#
# The comparison the appendix makes is deliberately narrow. Both arms inject a vector
# into the SAME Qwen3-4B under the SAME prompt and differ only in where the vector
# comes from: a doc2lora idea gene, or a mean-pooled hidden state read out of the
# frozen model and patched into a fixed position of a completion prompt (activation
# patching, Ghandeharioun et al. / Chen et al.). Everything is scored against what
# that prompt produces with no vector inserted, because the prompt has a strong prior
# of its own and an absolute number would mostly measure it.
#
# Three questions, three chains, run per arm and then scored together:
#
#   identification  does the decode name the SOURCE document, against 99 distractors
#                   drawn from the same PACS subdivision (decode_compare -> score_compare)
#   midpoints       does the midpoint of two vectors gain similarity to BOTH sources
#                   (midpoint_compare -> midpoint_score)
#   node means      do the 28 PACS node means decode to the right field
#                   (arith_decode -> arith_name -> arith_m3 / arith_m4)
#
# act_config selects the activation arm's layer/slots/mode; layer_sweep is what picks
# them, and the appendix reports block 15 with eight slots on a continuation prompt.
# Asking a direct question ("what is the topic?") returns empty strings, which is why
# the sweep exists rather than a guess.
#
# The naming and judging steps call OpenRouter, so they cost money and sit behind
# `actpatch_judged` rather than `actpatch`. The manuscript states these numbers in
# prose; nothing here is \input-ed.
#
# RUN: snakemake actpatch -j1          # GPU decodes + SBERT scoring, no LLM calls
#      snakemake actpatch_judged -j1   # the above plus the naming/judging panel
from os.path import join as j

ACT = "exps/2026-09-18-gene-vs-activation"
ACT_RES = j(ACT, "results")
ACT_ARMS = ["gene", "act"]
ACT_ENV = ("set -a; source .env 2>/dev/null; set +a; "
           f"export HF_HOME={config.get('hf_home', 'data/agent_assets/hf_cache')} "
           "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; ")

wildcard_constraints:
    arm="|".join(ACT_ARMS),


# ── Configuration sweep: which layer, how many slots, mean or last token ──────
rule act_layer_sweep:
    input:
        script=j(ACT, "layer_sweep.py"),
        common=j(ACT, "act_common.py"),
    output:
        j(ACT_RES, "layer_sweep.json"),
    params:
        n=config.get("act_sweep_docs", 50),
    resources:
        gpu=1,
    shell:
        ACT_ENV + f"python {ACT}/layer_sweep.py --n {{params.n}} --out {{output}}"


# ── Identification: decode 500 abstracts per arm, then retrieve the source ───
rule act_decode:
    input:
        script=j(ACT, "decode_compare.py"),
        common=j(ACT, "act_common.py"),
    output:
        j(ACT_RES, "decodes_{arm}.json"),
    params:
        n=config.get("act_decode_docs", 500),
    resources:
        gpu=1,
    shell:
        ACT_ENV + f"python {ACT}/decode_compare.py --arm {{wildcards.arm}} "
        "--n {params.n} --out {output}"


# SBERT only: retrieval against same-subdivision distractors, fidelity over the
# no-patch floor, and the repetition count the appendix quotes.
rule act_decode_score:
    input:
        decodes=expand(j(ACT_RES, "decodes_{arm}.json"), arm=ACT_ARMS),
        script=j(ACT, "score_compare.py"),
    output:
        j(ACT_RES, "score_compare.json"),
    shell:
        f"python {ACT}/score_compare.py --arms {' '.join(ACT_ARMS)} --out {{output}}"


# ── Midpoints: does averaging two vectors gain similarity to both sources ────
rule act_midpoints:
    input:
        script=j(ACT, "midpoint_compare.py"),
        common=j(ACT, "act_common.py"),
    output:
        j(ACT_RES, "midpoints_{arm}.json"),
    params:
        n=config.get("act_midpoint_pairs", 50),
    resources:
        gpu=1,
    shell:
        ACT_ENV + f"python {ACT}/midpoint_compare.py --arm {{wildcards.arm}} "
        "--n {params.n} --out {output}"


# The same scorer the T2L fusion arm uses (t2l.smk:t2l_fusion_score), so the two
# baselines are compared on one protocol rather than two.
rule act_midpoint_score:
    input:
        midpoints=expand(j(ACT_RES, "midpoints_{arm}.json"), arm=ACT_ARMS),
        script=j(ACT, "midpoint_score.py"),
    output:
        j(ACT_RES, "midpoint_score.json"),
    shell:
        f"python {ACT}/midpoint_score.py --arms {' '.join(ACT_ARMS)} "
        f"--decode-dir {ACT_RES} --out {{output}}"


# ── PACS node means: decode the 28 node means in both representations ────────
rule act_nodes:
    input:
        script=j(ACT, "arith_decode.py"),
        common=j(ACT, "act_common.py"),
    output:
        j(ACT_RES, "arith_nodes_{arm}.json"),
    resources:
        gpu=1,
    shell:
        ACT_ENV + f"python {ACT}/arith_decode.py --arm {{wildcards.arm}} --out {{output}}"


# Names each decoded continuation with an LLM that is NOT on the judge panel, so the
# namer never grades its own output. Costs OpenRouter calls.
rule act_node_names:
    input:
        nodes=expand(j(ACT_RES, "arith_nodes_{arm}.json"), arm=ACT_ARMS),
        script=j(ACT, "arith_name.py"),
    output:
        j(ACT_RES, "arith_m1.json"),
    shell:
        "set -a; source .env 2>/dev/null; set +a; "
        f"python {ACT}/arith_name.py --out {{output}}"


rule act_node_judge:
    input:
        names=j(ACT_RES, "arith_m1.json"),
        script=j(ACT, "arith_{metric}.py"),
    output:
        j(ACT_RES, "arith_{metric}.json"),
    wildcard_constraints:
        metric="m3|m4",
    shell:
        "set -a; source .env 2>/dev/null; set +a; "
        f"python {ACT}/arith_{{wildcards.metric}}.py --out {{output}}"


# ── Targets ──────────────────────────────────────────────────────────────────
rule actpatch:
    input:
        j(ACT_RES, "layer_sweep.json"),
        j(ACT_RES, "score_compare.json"),
        j(ACT_RES, "midpoint_score.json"),
        expand(j(ACT_RES, "arith_nodes_{arm}.json"), arm=ACT_ARMS),


rule actpatch_judged:
    input:
        rules.actpatch.input,
        j(ACT_RES, "arith_m3.json"),
        j(ACT_RES, "arith_m4.json"),
