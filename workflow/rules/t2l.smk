# Text-to-LoRA (Charakorn et al., ICML 2025) as a hypernetwork-adapter baseline (#151).
#
# The base method was rejected at ICLR 2026 partly for having no hypernetwork-adapter
# baseline, so this arm is not optional. The question is not whether T2L is a good
# hypernetwork -- it is whether GENERATING adapters is by itself enough to give a
# decodable space, or whether the space has to be learned.
#
# T2L reads a task description with a FROZEN gte-large encoder and expands the resulting
# vector into a LoRA. Three positions in that pipeline can be called "the embedding", and
# all three are run, because concluding "no space" from the input vector alone is exactly
# the shortcut a reviewer would find:
#
#   E0  the frozen gte-large output                        (1,024)
#   E1  the TaskEncoder output, Linear(1024,64)+LayerNorm   (   64)  <- the ONLY
#       document-dependent learned layer in the whole pipeline
#   E2  the generated LoRA factors, A and B averaged separately so the rank stays 8
#       (3,407,872) -- the structural analogue of doc2lora, whose head is linear
#
# Input policy: every method receives the paper's title+abstract
# VERBATIM. No task-description rewrite for T2L. The consequence -- T2L runs outside its
# training distribution -- is a stated interpretation limit, not something to patch
# experimentally. Claim ceiling: "as a document embedding, on the same input every other
# method gets, T2L's pipeline does not yield a decodable space." Nothing stronger.
#
# Generator is mistralai/Mistral-7B-Instruct-v0.2, the same base as the ICAE arm
# (config.yaml:icae_base_model), so decodes are comparable at the generator level.
#
# Runs in the MAIN env: hypermod.pt is a plain state_dict, so T2L's transformers==4.46.2
# / vllm==0.5.4 pins are only needed for their training harness. inflect, torchmetrics
# and wandb are required by hyper_llm_modulator's import chain.
#
# RUN: snakemake t2l_all -j1
import os
from os.path import join as j

T2L = "exps/2026-09-18-t2l-baseline"
T2L_RES = j(T2L, "results")
BT = "exps/2026-06-11-baseline-trees"
T2L_SRC = config.get("t2l_src", "text-to-lora/src")
T2L_SPACES = ["gte", "hidden", "dw"]          # E0 / E1 / E2
T2L_FUSION_SPACES = ["e0", "e1", "e2"]
T2L_CAP = config.get("t2l_label_cap", 2000)   # members averaged per node; matches
                                              # doc2lora, NOT ICAE's K=8 -- gte is cheap,
                                              # and a baseline run on 8 documents reads
                                              # as a baseline that was not really run.
T2L_ENV = (f"export T2L_SRC={T2L_SRC} PYTHONPATH={T2L_SRC} "
           f"HF_HOME={os.path.abspath(config.get('hf_home', 'data/agent_assets/hf_cache'))} "
           f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;")


# ── Pipeline-validity checks. Nothing downstream is reportable until these pass ──
#
# t2l_sanity : (a) do different documents give different adapters at all, and
#              (b) does a single document's adapter decode back to its own topic
# t2l_validity: does the pipeline reproduce T2L's INTENDED behaviour. The task must be
#              one the hypernetwork was TRAINED on -- args.yaml lists 479 and gsm8k is
#              not among them, and on a held-out task the adapter suppresses
#              chain-of-thought (the LoL tasks are short-answer), which looks like harm
#              and is not. lol_636 is in train_ds_names.
rule t2l_wiring:
    input: script=j(T2L, "verify_wiring.py"), common=j(T2L, "t2l_common.py"),
    output: j(T2L_RES, "verify_wiring.json"),
    resources: gpu=1,
    shell: T2L_ENV + " cd " + T2L + " && python verify_wiring.py"

rule t2l_sanity:
    input: script=j(T2L, "sanity_degeneracy.py"), common=j(T2L, "t2l_common.py"),
    output: j(T2L_RES, "sanity_degeneracy.json"),
    params: n_docs=config.get("t2l_sanity_docs", 60),
            n_desc=config.get("t2l_sanity_descs", 12),
    resources: gpu=1,
    shell: T2L_ENV + " cd " + T2L + " && python sanity_degeneracy.py "
           "--n-docs {params.n_docs} --n-desc {params.n_desc}"

rule t2l_validity:
    input: script=j(T2L, "task_validity.py"), common=j(T2L, "t2l_common.py"),
    output: j(T2L_RES, "task_validity_lol_636.json"),
    params: task="lol_636", n=config.get("t2l_validity_n", 80),
    resources: gpu=1,
    shell: T2L_ENV + " cd " + T2L + " && python task_validity.py "
           "--task {params.task} --n {params.n}"


# ── Cluster-label arm: one {code: label} JSON per embedding position ──────────
rule t2l_label:
    input: script=j(T2L, "t2l_label.py"), common=j(T2L, "t2l_common.py"),
           prompt=j(BT, "shared_prompt.py"),
           validity=j(T2L_RES, "task_validity_lol_636.json"),
    # The resume state is deliberately NOT declared as an output: snakemake clears a
    # rule's outputs before re-running it, which wiped the state file and forced all 28
    # nodes through the full CAP=2000 pass again (~42 min) even though every one of them
    # was already done. t2l_fusion keeps its state for the same reason.
    output: expand(j(T2L, "t2l_{space}_labels.json"), space=T2L_SPACES),
    params: cap=T2L_CAP,
    resources: gpu=1, mem_gb=40,
    shell: T2L_ENV + " cd " + T2L + " && python t2l_label.py --cap {params.cap}"

# label_eval_prep.py resolves METHOD_FILES relative to the baseline-trees dir
rule t2l_label_register:
    input: expand(j(T2L, "t2l_{space}_labels.json"), space=T2L_SPACES),
    output: expand(j(BT, "t2l_{space}_labels.json"), space=T2L_SPACES),
    shell: "cp {input} " + BT + "/"


# ── Fusion arm: the same 100 corner pairs as §4.3 (50 near L1 + 50 far L5) ────
# The prediction is that T2L midpoints do NOT blend, because its coordinates belong to a
# frozen encoder. If they DO blend, adapter-space smoothness is a general property of
# hypernetwork adapters and §5's mechanism paragraph has to be rewritten -- which is why
# this is worth running either way.
rule t2l_fusion:
    input: script=j(T2L, "t2l_fusion.py"), common=j(T2L, "t2l_common.py"),
           corners=expand("exps/2026-07-02-simplex-kwgrid/corners_pair{L}_{n}.json",
                          L=["L1", "L5"], n=[f"{i:02d}" for i in range(50)]),
    output: expand(j(T2L_RES, "midpoints_t2l_{sp}.json"), sp=T2L_FUSION_SPACES),
    resources: gpu=1,
    shell: T2L_ENV + " cd " + T2L + " && python t2l_fusion.py --n 50"

# scored by the gene-vs-activation scorer (#152) so the arms are directly comparable
rule t2l_fusion_score:
    input: expand(j(T2L_RES, "midpoints_t2l_{sp}.json"), sp=T2L_FUSION_SPACES),
           script="exps/2026-09-18-gene-vs-activation/midpoint_score.py",
    output: j(T2L_RES, "t2l_fusion_score.json"),
    shell: "python {input.script} --arms " + " ".join(f"t2l_{s}" for s in T2L_FUSION_SPACES)
           + " --decode-dir " + T2L_RES + " --out {output}"


rule t2l_all:
    input:
        j(T2L_RES, "verify_wiring.json"),
        j(T2L_RES, "sanity_degeneracy.json"),
        j(T2L_RES, "task_validity_lol_636.json"),
        expand(j(BT, "t2l_{space}_labels.json"), space=T2L_SPACES),
        j(T2L_RES, "t2l_fusion_score.json"),
