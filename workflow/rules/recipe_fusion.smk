# Recipe-fusion demo (paper App. recipe-fusion). Mistral-7B doc2lora checkpoint;
# an Italian (cacio e pepe) and a Japanese (kake udon) recipe, full-rank renorm
# midpoint interpolation. The GPU rule regenerates the transcript JSON; the CPU
# assembler emits the verbatim source/midpoint blocks the appendix quotes.
from os.path import join as j

RECIPE_DIR  = j("exps", "2026-06-11-recipe-fusion")
RECIPE_JSON = j(RECIPE_DIR, "results", "recipe_fusion.json")
RECIPE_TEX  = j(config.get("figs_dir", "figs"), "recipe_fusion.tex")

rule recipe_fusion_decode:
    output:
        RECIPE_JSON,
    params:
        checkpoint=MISTRAL_CHECKPOINT_PATH,
    resources:
        gpu=1,
    script:
        "../../workflow/scripts/recipe_fusion.py"

rule tab_recipe_fusion:
    input:
        json=RECIPE_JSON,
        script="workflow/plot/fig_recipe_fusion.py",
    output:
        table=RECIPE_TEX,
    script:
        "../plot/fig_recipe_fusion.py"

rule recipe_fusion_all:
    input:
        RECIPE_TEX,
