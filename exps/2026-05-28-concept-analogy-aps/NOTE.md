# 2026-05-28 — Concept Analogy on APS

Can doc2lora **name a cluster of papers**, and does its embedding space encode the
**PACS hierarchy** (broad concept → subconcept → sub-subconcept)?

Idea: take a set of papers, average their `[26, 8, 512]` perceiver latents, internalize
the average as LoRA weights, and ask the model to name the shared topic. Compare the
generated label to the cluster's canonical APS/PACS subject, and study where labels and
centroids sit in the embedding geometry.

## Hierarchy (4 levels, broad → specific)
- **main** — 9 APS subject classes, human titles (`category_table.csv`): "Condensed matter", …
- **division** — PACS 2-digit, e.g. `74` → "Superconductivity" (`scripts/pacs_scheme.py`)
- **subdivision** — PACS `XX.YY`, e.g. `74.20`
- **specific** — full PACS1 code, e.g. `74.20.-z`

PACS levels nest definitionally by code prefix; each division is also linked to its
majority APS main class for a readable top.

## Pipeline (`scripts/`, chained by `Snakefile`)
1. **build_groups.py** (CPU) — assign 418k papers (with PACS1 + embedding + ≥300-char text)
   to the 4 levels, attach canonical label text, build the tree. → `paper_groups.parquet`,
   `groups.parquet`, `tree.json`.
2. **decode_groups.py** (GPU) — per group, average K=20 centroid-nearest papers' full latents,
   decode a label + description; also embed the generated and canonical labels. →
   `decoded_labels.parquet`, `group_embeddings.npz`, cached `full_embeddings/`.
3. **eval_semantic_match.py** (HOST, Ollama `minimax-m2.7:cloud`) — LLM judge scores
   generated vs canonical subject (1 / 0.5 / 0; a correct subfield counts as match). →
   `semantic_match.parquet`, `semantic_match_summary.json`.
4. **hierarchy_geometry.py** (CPU) — true centroids over ALL members; tests
   parent≈mean-of-children, child→parent assignment, radial structure (central vs
   peripheral), tree recovery (cophenetic corr), and where labels land. →
   `centroids.npz`, `hierarchy_metrics.json`, `radial_by_level.csv`.
5. **concept_analogy_walkthrough.py** — marimo notebook (interactive).

## How to run (doc2lora container = `/workspace`)
```bash
# step 1 (CPU)
docker exec doc2lora bash -lc 'cd /workspace && python exps/2026-05-28-concept-analogy-aps/scripts/build_groups.py'
# step 2 (GPU) — tiny first, then full
docker exec doc2lora bash -lc 'cd /workspace && CUDA_VISIBLE_DEVICES=0 python exps/2026-05-28-concept-analogy-aps/scripts/decode_groups.py --levels main --k 10'
docker exec doc2lora bash -lc 'cd /workspace && CUDA_VISIBLE_DEVICES=0 python exps/2026-05-28-concept-analogy-aps/scripts/decode_groups.py'
# step 3 (HOST — ollama)
python3 exps/2026-05-28-concept-analogy-aps/scripts/eval_semantic_match.py
# step 4 (CPU, in container for /workspace paths)
docker exec doc2lora bash -lc 'cd /workspace && CUDA_VISIBLE_DEVICES="" python exps/2026-05-28-concept-analogy-aps/scripts/hierarchy_geometry.py'
```

## Findings (2026-05-28, 609 decoded groups; geometry over 418k papers)

**1. doc2lora can name a cluster.** Averaging cluster latents and decoding gives coherent
topics. LLM-judge (minimax-m2.7:cloud) mean score by level:

| level | mean | exact-match | n |
|---|---|---|---|
| main | 0.89 | 0.78 | 9 |
| division | 0.74 | 0.54 | 68 |
| subdivision | 0.75 | 0.54 | 323 |
| specific | 0.81 | 0.64 | 200 |

The mid-level dip is **largely an eval artifact**: the model emits a *specific subfield*
while the canonical division/subdivision text is broad, and our PACS subdivision
descriptions are sparse (coarse fallback). e.g. PACS `05.70.Ln` GT fell back to
"Thermodynamics" but is really "Nonequilibrium and irreversible thermodynamics" — doc2lora
said "Nonequilibrium statistical mechanics" (correct) yet was scored 0.5. Enrich
`pacs_scheme.SUBDIVISIONS` to firm this up.

**2. The space is a radial onion — broad concepts central, specific peripheral.**
Mean distance to the global centroid grows monotonically: main 14.2 → division 17.2 →
subdivision 20.5 → specific 22.3 (cos to global 0.971 → 0.957 → 0.942 → 0.933).

**3. Parent concept ≈ mean of its children.** cos(parent, mean-of-true-children) vs random:
main←div 0.995/0.965, div←sub 0.996/0.945, sub←spec 0.998/0.909. Real ≫ random; gap widens
with depth (additive/compositional structure).

**4. Children point to their true parent.** child→parent nearest-centroid top-1:
div→main 75% (chance 11%), sub→div 74% (1.5%), spec→sub 79% (0.18%). 7×–430× chance.
Margins are thin (anisotropic, high-cosine space) but ranking is right ~3/4 of the time.

**5. Global tree recovery is moderate.** Agglomerative on subdivision centroids: cophenetic
corr 0.65, but vs the "same-PACS-division" grouping only 0.22 — local nesting strong, full
tree blurs (many subfields are cross-divisional).

**6. Labels are semantically right but geometrically displaced.** Re-embedding the
generated/canonical *label string* lands far from the cluster it names:
cos(label, own centroid) ≈ 0.33 at every level; label retrieves its own centroid #1 only
3.6% of the time (~22× chance). doc2lora encodes *documents* — a 3-word topic is a degenerate
document, so the encode direction (label→latent) is NOT aligned with the document-cluster
centroid, even though the decode direction (latent→label) is excellent.

## Multi-document fusion in perceiver latent space (2026-05-28 exploration)

We explored whether averaging perceiver latents across documents recovers the **parent/umbrella concept**, and which level and method works best.

### Spaces

The perceiver has two stages:
- **Encoder** (`perceiver.encoder`): cross-attention resampler; each document token sequence → `[26, 8, 512]` (n_LM_layers × n_latent_queries × hidden_dim). Float32.
- **Decoder** (`perceiver.decoder`): self-attention on the 8 latent slots; `[26, 8, 512]` → `[26, 8, 512]`. Refines the encoder representation.

### Fusion strategies compared

| Strategy | How | Decoded broad label |
|---|---|---|
| FiD (encoder-level concat) | encode each doc independently, concat ctx features, run perceiver jointly | specific subfield (first-doc bias) |
| Slot fusion | per-doc encoder mean → `[26,512]`; stack N docs as N slots → tile to `[26,8,512]` | correct umbrella ✓ |
| Mean pool → tile | average N encoder outputs → `[26,512]` → tile `[26,8,512]` | correct umbrella ✓ |
| Mean of decoder outputs | per-doc: encoder → decoder; average decoder outputs; inject after decoder | same as mean pool ✓ |

### Key findings

**1. Encoder-space mean is geometrically closer to parent than any child.**
cos(parent, mean_children_enc) ≈ 0.83–0.84 vs avg child→parent 0.68–0.72. The centroid in encoder space pulls toward the parent region — this is *why* decoding the mean recovers the umbrella label.

**2. Decoder-space mean is geometrically tighter but does not improve decoded text.**
cos(parent_dec, mean_children_dec) ≈ 0.87–0.88 (higher than encoder), but decoded text is identical. The decoder output space is not convex for simple averaging: averaging decoder outputs then re-running the decoder produces "Social Sciences" (off-manifold). Injecting after the decoder (bypassing second pass) recovers correct labels but with no improvement over encoder-space mean.

**3. Slot fusion and mean pool collapse to the same decoder output.**
In encoder space: sim(slot_fused, mean_tiled) ≈ 0.86. After decoder self-attention: sim(slot_dec, mean_dec) ≈ 0.97. The decoder self-attention collapses diverse slots toward the same centroid as identical-slot mean pool. Decoded text is identical in all cases.

**4. Mean pool is the right fusion method — simpler and slightly better.**
Mean-tiled is consistently closer to parent than slot-fused in both encoder and decoder space. Slot fusion adds no value once the decoder runs.

**5. FiD does NOT compute the centroid.**
cos(FiD, parent) ≈ 0.73; cos(FiD, mean_children) ≈ 0.93. FiD anchors to child-space, biased by whichever tokens dominate cross-attention.

### Idea: fusion via decoder slot (set intersection hypothesis)

**Motivation.** The perceiver decoder applies self-attention across the 8 latent slots. If each slot carries a *different document's* representation, the self-attention could in principle compute what is *shared* across all documents — a geometric set intersection — landing the output near the parent concept.

**Tested as (C) in `test_encoder_vs_decoder_space.py`:**
- Per-doc encoder mean → `[26, 512]` each; stack N children as N slots → `[26, N, 512]` → tile to `[26, 8, 512]`; run through decoder.
- Compare sim(slot_dec, parent_dec) vs sim(mean_dec, parent_dec).

**Result: hypothesis fails geometrically.**
slot_dec ↔ parent_dec ≈ 0.80; mean_dec ↔ parent_dec ≈ 0.82. Slot fusion through the decoder is *less* close to parent than encoder-space mean. And slot_dec ↔ mean_dec ≈ 0.97 — the decoder collapses both approaches to nearly the same point regardless.

**Why it fails.** The decoder self-attention mixes the 8 slots but doesn't extract their intersection — it averages them (attention weights sum to 1, and with diverse inputs the result is a weighted centroid, not an intersection). The "set intersection" would require something like a min-pooling or contrastive operation, which cross/self-attention doesn't natively implement. Encoder-space mean achieves the centroid directly and cheaply.

### Practical recipe
```python
# For N documents, recover umbrella label:
enc_lats = [extract_full_latents(model, ctx_tok, doc) for doc in docs]   # [26,8,512] each
mean_lat  = torch.stack(enc_lats).float().mean(dim=0)                     # [26,8,512]
internalize_from_latents(model, mean_lat.to(model.device))
label = ask(model, gen_tok, "What broader research field do all topics belong to? One field name only.")
```

### Relevant scripts
- `scripts/test_centroid_hypothesis.py` — verifies mean_enc is geometrically closer to parent than children or FiD
- `scripts/test_encoder_vs_decoder_space.py` — compares enc vs dec space geometry; slot-fusion set-intersection test
- `scripts/avg_decoder_queries.py` — averages decoder outputs, injects after decoder (no second pass)
- `scripts/slot_vs_mean.py` — slot fusion vs mean pool in both encoder and decoder space

## Key reuse
`../2026-05-24-embedding-exploration/scripts/utils.py` — `load_model`,
`extract_full_latents`, `internalize_from_latents`, `ask` (the decode path).

## Data
- `data/aps/embeddings/gemma_mean_pooled.npz` — 644,022 × 13,312 fp16
- `data/aps/paper_text.parquet`
- `/data/datasets/aps/preprocessed/{paper_table,paper_category_table,category_table}.csv`
- checkpoint: `/opt/doc-to-lora/trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin`
