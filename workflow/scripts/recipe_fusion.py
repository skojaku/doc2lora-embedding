"""[GPU] Recipe-fusion demo for the Decoding subsection (paper SI).

Embeds an Italian recipe (cacio e pepe) and a Japanese recipe (kake udon),
then (1) verifies the in-paper claim that the Italian adapter answers
"What cheese is used?" with "Pecorino Romano" without the recipe in context,
and (2) decodes the renorm((1-alpha)*V_A + alpha*V_B) interpolation at
alpha in {0.25, 0.5, 0.75} with a neutral recipe prompt to show the fusion.

Run (GPU):
  CUDA_VISIBLE_DEVICES=0 python workflow/scripts/recipe_fusion.py

or through the workflow, which passes the checkpoint from workflow/config.yaml:
  snakemake recipe_fusion -j1
"""
import json
import os
from pathlib import Path

import torch

EXP_DIR = Path(__file__).resolve().parent
from bench_data import out_dir   # bench_data.py sits next to this file
DATA = out_dir("recipe_fusion")           # where this chain writes

REPO = EXP_DIR.parents[1]  # data/recipe_fusion -> repo root

# The frozen legacy API (libs/legacy, installed by install.sh) carries everything
# this demo needs: load the hypernetwork, read a document into an embedding,
# interpolate two of them, inject the result through the linear head, and decode.
from doc2lora_legacy import (  # noqa: E402
    load_model, extract_norm_lora_emb, interpolate_embeddings,
    internalize_from_norm_lora_emb, generate_text,
)

# Mistral-7B doc2lora checkpoint; override via $DOC2LORA_MISTRAL_CKPT or the
# workflow's mistral_checkpoint_path.
CHECKPOINT = os.environ.get(
    "DOC2LORA_MISTRAL_CKPT",
    str(REPO / "data/agent_assets/mistral_7b_d2l/checkpoint-20000/pytorch_model.bin"),
)


def decode_from_emb(model, gen_tok, emb, prompt, max_new_tokens=128):
    """Inject one embedding through the linear head and decode under `prompt`.

    Resets first, so a decode never inherits the previous document's adapter.
    Greedy decoding (generate_text's default), so the transcript is deterministic.
    """
    model.reset()
    internalize_from_norm_lora_emb(model, emb)
    return generate_text(model, gen_tok, prompt, max_new_tokens=max_new_tokens).strip()

ITALIAN = """Cacio e Pepe (Roman pasta with cheese and black pepper)

A classic recipe of the Roman tradition with only three ingredients. Serves 2.

Ingredients:
- 200 g spaghetti (or tonnarelli)
- 100 g Pecorino Romano cheese, finely grated
- 2 teaspoons whole black peppercorns
- coarse salt for the pasta water

Preparation:
1. Toast the black peppercorns in a dry skillet over medium heat until fragrant,
   then crush them coarsely in a mortar.
2. Bring a pot of lightly salted water to a boil and cook the spaghetti until
   one minute short of al dente. Use less water than usual so the cooking water
   becomes rich in starch.
3. While the pasta cooks, mix the grated Pecorino Romano with a ladle of the hot,
   starchy pasta water, stirring vigorously until it forms a thick, smooth cream
   with no lumps.
4. Transfer the spaghetti to the skillet with the crushed pepper, add a splash of
   pasta water, and finish cooking for one minute, tossing constantly.
5. Off the heat, add the Pecorino cream and toss vigorously, loosening with more
   pasta water as needed, until every strand is coated in a glossy sauce.
6. Serve immediately with extra grated Pecorino and a final grind of pepper."""

JAPANESE = """Kake Udon (Japanese udon noodles in hot dashi broth)

A simple and warming Japanese noodle soup built on dashi stock. Serves 2.

Ingredients:
- 2 portions fresh or frozen udon noodles
- 600 ml dashi stock (kombu and katsuobushi)
- 3 tablespoons soy sauce
- 2 tablespoons mirin
- 1 teaspoon sugar
- 2 spring onions, thinly sliced
- shichimi togarashi (Japanese seven-spice), to taste
- optional toppings: kamaboko fish cake, tempura flakes (tenkasu), a soft egg

Preparation:
1. Make the dashi: soak a piece of kombu in cold water for 30 minutes, bring it
   almost to a boil and remove the kombu, then add a handful of katsuobushi
   (dried bonito flakes), turn off the heat, let it steep for 2 minutes, and strain.
2. Season the dashi with the soy sauce, mirin, and sugar, and keep the broth hot
   over low heat without boiling.
3. Cook the udon noodles in a separate pot of unsalted boiling water according to
   the package directions, then drain and rinse briefly to remove surface starch.
4. Divide the noodles between two deep bowls and ladle the hot seasoned broth
   over them.
5. Top with sliced spring onions and any optional toppings, and finish with a
   pinch of shichimi togarashi. Serve immediately, very hot."""

RECIPE_PROMPT = ("Write out this recipe: give the dish a name, list the "
                 "ingredients, and describe the preparation steps.")
CHEESE_PROMPT = "What cheese is used?"
ALPHAS = [0.25, 0.5, 0.75]


def main(out_path=None, checkpoint=CHECKPOINT):
    model, gen_tok, ctx_tok = load_model(checkpoint, mode="full")
    emb_a = extract_norm_lora_emb(model, ctx_tok, ITALIAN)
    emb_b = extract_norm_lora_emb(model, ctx_tok, JAPANESE)
    print("extracted embeddings:", tuple(emb_a.shape), flush=True)

    out = {
        "checkpoint": CHECKPOINT,
        "recipe_prompt": RECIPE_PROMPT,
        "cheese_prompt": CHEESE_PROMPT,
        "doc_italian": ITALIAN,
        "doc_japanese": JAPANESE,
        "decodes": {},
    }

    # (1) The in-paper claim: QA against the Italian adapter alone.
    out["decodes"]["italian_cheese_qa"] = decode_from_emb(
        model, gen_tok, emb_a, CHEESE_PROMPT, max_new_tokens=40)
    print("\n[A] cheese QA:", out["decodes"]["italian_cheese_qa"], flush=True)

    # (2) Endpoints and interpolations under the neutral recipe prompt.
    out["decodes"]["alpha_0.00_italian"] = decode_from_emb(
        model, gen_tok, emb_a, RECIPE_PROMPT, max_new_tokens=450)
    print("\n[A] recipe decode:\n", out["decodes"]["alpha_0.00_italian"], flush=True)

    for alpha in ALPHAS:
        mid = interpolate_embeddings(emb_a, emb_b, alpha, mode="full", renorm=True)
        key = f"alpha_{alpha:.2f}"
        out["decodes"][key] = decode_from_emb(
            model, gen_tok, mid, RECIPE_PROMPT, max_new_tokens=450)
        print(f"\n[alpha={alpha}] decode:\n", out["decodes"][key], flush=True)

    out["decodes"]["alpha_1.00_japanese"] = decode_from_emb(
        model, gen_tok, emb_b, RECIPE_PROMPT, max_new_tokens=450)
    print("\n[B] recipe decode:\n", out["decodes"]["alpha_1.00_japanese"], flush=True)

    out_path = Path(out_path) if out_path else DATA / "recipe_fusion.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nsaved to", out_path, flush=True)


if "snakemake" in globals():
    main(out_path=snakemake.output[0],  # noqa: F821
         checkpoint=snakemake.params.get("checkpoint", CHECKPOINT))  # noqa: F821
elif __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    args = ap.parse_args()
    main(out_path=args.out, checkpoint=args.checkpoint)
