# Recipe fusion demo for the paper's Decoding subsection (SI)

## Question
The Decoding subsection of the ICLR paper (`paper/iclr2026/main.tex`,
`sec:decode`) uses two recipe claims as its running example: (1) an adapter
made from an Italian recipe answers "What cheese is used?" with "Pecorino
Romano" without the recipe in context, and (2) interpolating an Italian
recipe with a Japanese recipe yields a fusion recipe. Are both claims true
when actually run?

## Method
`recipe_fusion.py`, run inside the `doc2lora` container on aster (GPU 3),
Mistral-7B doc2lora (`/opt/doc-to-lora/trained_d2l/mistral_7b_d2l/checkpoint-20000`),
greedy decoding:

1. Two ~250-word source documents written for this demo: **cacio e pepe**
   (spaghetti, Pecorino Romano, black pepper) and **kake udon** (udon,
   kombu/katsuobushi dashi, soy, mirin). Extract `norm_lora_emb` per doc.
2. QA check on the Italian adapter alone: "What cheese is used?" (40 tokens).
3. Decode endpoints and `renorm((1-a)*V_A + a*V_B)` at a in {0.25, 0.5, 0.75}
   (`interpolate_embeddings`, `mode="full"`) under the neutral prompt
   "Write out this recipe: give the dish a name, list the ingredients, and
   describe the preparation steps." (450 tokens).

## Findings
- **QA claim verified:** the Italian adapter answers "The cheese used in this
  recipe is Pecorino Romano."
- **Midpoint (a=0.5) is a genuine fusion:** "Kake Udon with Creamy Parmesan
  Sauce" — udon noodles coated in a melted-cheese cream sauce (the Italian
  technique) seasoned with mirin, soy sauce, and wasabi, topped with spring
  onion and black pepper. Neither source contains a cheese-sauced udon.
- **Endpoints are faithful:** a=0 reproduces cacio e pepe, a=1 reproduces
  kake udon (small paraphrases/additions, e.g. olive oil at a=0).
- **Off-midpoint decodes snap to the nearer source:** a=0.25 is still cacio e
  pepe; a=0.75 is mostly kake udon with butter creeping into the broth.
- **Caveats:** the midpoint substitutes Parmesan for Pecorino Romano and
  introduces heavy cream/butter found in neither source — the fusion is a
  plausible cuisine blend, not a constraint-satisfying merge of the two
  ingredient lists. Single greedy run, curated pair.

## Key files
- `recipe_fusion.py` — the demo script (source docs inline).
- `results/recipe_fusion.json` — source docs, prompts, all decodes.
- `results/run.log` (local) / `run.log` (aster) — full transcript.
- Paper: SI section `app:recipe-fusion` in `paper/iclr2026/main.tex`,
  referenced from `sec:decode`.

---

**README index summary:** Runs the Decoding-subsection recipe example for the
paper SI on Mistral-7B doc2lora: a cacio e pepe adapter answers "What cheese
is used?" with Pecorino Romano (no context), and the a=0.5 interpolation with
a kake udon recipe decodes to "Kake Udon with Creamy Parmesan Sauce", a real
fusion applying the Italian cheese-sauce technique to Japanese ingredients;
a=0.25/0.75 snap to the nearer parent. Outputs in results/recipe_fusion.json,
quoted verbatim in SI app:recipe-fusion.
