# Abstraction walk (radius -> generality) behind Fig. 2 (d) and App. D.3 length-dial.
# Embed one Wikipedia lead (wiki_leads.json), scale its representation toward 0, and
# decode at each radius. doc2lora abstracts upward (specific -> field -> base prior);
# ICAE is magnitude-invariant; vec2text holds the source specificity then degrades.
#
# ICAE gets the SAME decode prompt as doc2lora (like-for-like); vec2text is an
# inverter (no prompt). HEAVY: GPU + qwen checkpoint (doc2lora), ICAE weights
# (icae), and the isolated .venv-vec2text. Dry-run validated only.
from os.path import join as j

AW_DIR = "data/labels"
AW_GPU = config.get("abstraction_walk_gpu", "0")
AW_VEC2TEXT_PY = config.get("vec2text_python", ".venv-vec2text/bin/python")

AW_LEADS = j(SCRIPTS, "wiki_leads.json")   # the seed documents, not an output
AW_DOC2LORA = j(AW_DIR, "abstraction_walk.json")
AW_ICAE = j(AW_DIR, "icae_abstraction_walk.json")
AW_VEC2TEXT = j(AW_DIR, "vec2text_abstraction_walk.json")


# leads + script are git-tracked and the output ships in the `results` tier (Fig. 2
# panel (d) reads it); a fresh checkout gives them arbitrary mtimes, so they are ancient()
# -- otherwise `snakemake fig2` on a clone with the data symlink would re-run
# this GPU decode just to refresh a file it already has.
rule aw_doc2lora:
    input:
        leads = ancient(AW_LEADS),
        ckpt = QWEN_CHECKPOINT_PATH,
        script = ancient(j(SCRIPTS, "abstraction_walk.py")),
    output:
        result = AW_DOC2LORA,
    resources:
        gpu = 1,
    shell:
        "CUDA_VISIBLE_DEVICES={AW_GPU} python {input.script}"


# Same ancient() reasoning as aw_doc2lora: the output is a fetched artifact, so a clone
# must not re-run a GPU decode over mtimes alone. The ICAE checkpoint is NOT an input --
# icae_label.py resolves it from config.yaml:icae_weights at run time, exactly as the other
# ICAE rules do (bt_icae_raw, kg_decode_icae, fig2_decode_curated_icae). Declaring it here
# made this rule the only one that could not even be dry-run without the third-party
# weights unpacked, which broke `snakemake -n abstraction_walk_all`.
rule aw_icae:
    input:
        leads = ancient(AW_LEADS),
        helper = ancient(j(SCRIPTS, "icae_label.py")),
        script = ancient(j(SCRIPTS, "icae_abstraction_walk.py")),
    output:
        result = AW_ICAE,
    resources:
        gpu = 1,
    shell:
        "CUDA_VISIBLE_DEVICES={AW_GPU} python {input.script}"


rule aw_vec2text:
    input:
        leads = AW_LEADS,
        script = j(SCRIPTS, "vec2text_abstraction_walk.py"),
    output:
        result = AW_VEC2TEXT,
    resources:
        gpu = 1,
    shell:
        "CUDA_VISIBLE_DEVICES={AW_GPU} {AW_VEC2TEXT_PY} {input.script}"


rule abstraction_walk_all:
    input:
        AW_DOC2LORA,
        AW_ICAE,
        AW_VEC2TEXT,
