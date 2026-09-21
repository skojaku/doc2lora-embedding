# Abstraction walk: doc2lora vs. ICAE vs. vec2text

Magnitude-shrink (radius) experiment behind §length-generality and
App.~length-dial. Embed one document, scale its representation toward 0, decode at
each radius, and ask whether the decoded concept *broadens* as the vector shrinks.

## Scripts / data

- Input: `wiki_leads.json` — 5 single Wikipedia leads (one per "entity type"):
  `physics concept` (fractional quantum Hall), `biochemistry` (citric acid cycle),
  `fruit` (banana), `famous person` (Nikola Tesla), `patent (LED)` (GaN LED).
- `abstraction_walk.py`        → `abstraction_walk.json`        (doc2lora-qwen, GPU)
- `icae_abstraction_walk.py`   → `icae_abstraction_walk.json`   (ICAE, GPU; restores `icae_label.py`)
- `vec2text_abstraction_walk.py` → `vec2text_abstraction_walk.json` (vec2text GTR, `.venv-vec2text`)

Shared `ALPHAS = [1.15, 0.90, 0.70, 0.55, 0.42, 0.30, 0.20, 0.12, 0.06]`.
ICAE receives the **same** decode prompt as doc2lora ("In two to four words, name
the topic or category…") for a like-for-like comparison; vec2text is an inverter
(no prompt). Norms live in different spaces and are not comparable across methods
(doc2lora ~17, ICAE ~2.7k, vec2text ~0.5); only the α multiplier matters.

## Findings

- **doc2lora — abstracts upward, then hits the base-model prior.** As α falls the
  label climbs a specific→general ladder before collapsing near the origin:
  - Tesla biography → "Nikola Tesla and AC power" → "Nuclear energy basics" →
    "Electricity and Safety" → "AI and Robotics"
  - banana → "Plant biology" → "Plant-based foods" → "AI and Robotics"
  - LED → "Semiconductor device structure" → "Semiconductor materials" → "AI…"
  - citric acid → "Cellular respiration" → "AI…"; quantum Hall → "Quantum physics"
    → "Quantum computing" → "AI…". Near-origin collapse to "AI and Robotics" is the
    decoder's prior and is not reliable (noted in the paper).

- **ICAE — magnitude-invariant.** Identical *specific* label at all 9 radii (its
  memory slots are L2-normalized internally, so scaling does nothing until they
  vanish). e.g. "Nikola Tesla Biography" at every α. Never abstracts.

- **vec2text — holds the source specificity, never abstracts, degrades to noise.**
  It keeps reconstructing the *same* entity at every radius (never broadens to a
  field); as α shrinks the reconstruction degrades into garbled / off-topic text
  (spurious URLs, citations, "neo-Geo/neo-Nazi" runs at low α). It does **not**
  collapse to a single fixed string (that was the molecular-biology run; it does
  not generalize here).

## Wording caveat for the paper

App.~length-dial's "vec2text stays coherent … collapses to a fixed string at β=0"
is only partly supported: "never abstracts / same specificity" holds, but it is
not coherent at low α and does not collapse to one fixed string. Honest phrasing:
*"vec2text holds the source's specificity at every radius and never abstracts to a
broader field; as the vector shrinks its reconstruction degrades into increasingly
garbled, off-topic text."*

## Reproduce

`snakemake abstraction_walk` (rules in `workflow/rules/abstraction_walk.smk`;
needs a GPU + qwen/ICAE checkpoints + `.venv-vec2text`).
