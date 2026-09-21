# General doc2lora kron adapter (one adapter from broad OpenAlex citations) — final results

## Method
Sampled 42,332 citation TRIPLETS (anchor, positive, easy-neg, hard-neg; SPECTER2-style) from the MASTER
OpenAlex edge list (/data/datasets/openalex: 125M papers, 2.56B edges), 167k-paper pool. Fresh doc2lora
embedding of the pool (gemma/qwen/mistral, lib mode=embed, max_tokens=512). Trained one Kron adapter per
encoder with in-batch InfoNCE + explicit hard+easy negatives, NO regularization (broad data). Applied the
single adapter to ALL benchmarks (next-paper/topic/collab field tasks + S2AND name disambiguation).
Snakemake: rule `general` (workflow/rules/general_adapter.smk). Scripts: sample_edges/embed_pool/train_general/apply_general.

## Findings
- FIELD TASKS: general genkron recovers ~all of each FIELD-SPECIFIC adapter's gain (within .02-.03), beats
  SPECTER2 on next-paper+topic (qwen econ .912/.373 > .905/.336), and TRANSFERS to APS (physics, unseen):
  np .937 (gemma) / .949 (qwen) / .960 (mistral) vs raw ~.85. Topic/collab: SBERT still leads, genes competitive.
- NAME DISAMBIGUATION (B3): the general CITATION adapter BEATS the SAME-AUTHOR-trained adapter on pubmed
  (+.05-.07 all 3 enc: .798/.815/.816 vs .74/.75/.78), kisti (all enc), arnetminer (gemma/qwen); ~ties qian.
  Beats SPECTER on 4/5. qwen arnetminer .708 > sbert .698. => broad citation signal generalizes to author-
  identity BETTER than in-task author-label training.
- Encoder pick: qwen best quality (field tasks + AND), gemma efficient default (~.01-.02 behind, 3x faster,
  2B), mistral only for niche wins (aps np .960-.970, pubmed AND .816) at 3-4x embed cost.

## Bottom line
ONE adapter trained on broad OpenAlex citations is the best GENE representation across benchmarks: matches/
beats per-task adapters, transfers to unseen domains+tasks, and is the best embedding for name disambiguation
(beats in-task adapter AND SPECTER). Per-bench winners: next-paper=genes+kron; AND=genes+general; topic/collab=SBERT.

---

## 2026-06-25 — Data-scaling test (qwen): does 2x training data help?

**Tried:** Doubled the general adapter's training data and steps for qwen, held everything else fixed.
Resampled the master-OpenAlex triplets at 2x (42,332 -> 85,635 triplets; pool 167,111 -> 335,085 papers),
freshly doc2lora-embedded the 252,450 new pool papers (left-pad fix for Qwen3+FlashAttn; reused the 82,635
already-embedded), retrained the Kron adapter at STEPS 8000 -> 16000. Re-applied and re-evaluated on
collab / next-paper / topic across economics, psychology, aps, plus the #65 interdisciplinary-retrieval
geometry test. (1x adapters/genes backed up as *_1x; 2x as *_2x; canonical genkron restored to 1x.)

**Findings:** No meaningful improvement anywhere. qwen_genkron, 1x -> 2x:

| field | collab | np | topic F1 | topic acc |
|---|---|---|---|---|
| economics | 0.626->0.624 | 0.906->0.909 | 0.353->0.358 | 0.743->0.744 |
| psychology | 0.690->0.687 | 0.918->0.918 | 0.360->0.371 | 0.719->0.727 |
| aps | 0.819->0.818 | 0.955->0.955 | 0.613->0.622 | 0.735->0.741 |

Every delta is within +-0.011: next-paper flat, collab a hair negative, topic a hair positive (no systematic
direction). #65 interdisc retrieval: lift@20 39.1x -> 38.7x (flat), mix-parent@20 +0.019 -> +0.023 (tiny),
balance ~0.09 unchanged. Training loss ended higher with 2x (more diverse data), as expected, but eval is flat.

**Learning:** The general adapter is already at its data-saturation point at 42k triplets / 8k steps — the
263k-param invertible Kron transform has captured what the broad citation signal can give it, and feeding it
2x more data (and 2x steps) buys nothing measurable. Consistent with the per-layer-kron finding that *breadth*
of supervision, not *volume*, is the lever. To move these numbers further, change the signal (richer negatives,
different supervision, more adapter capacity), not the data quantity.

**Gotcha:** Qwen3 + FlashAttention rejects right-padded batches (`padding_side='right'` ValueError); the legacy
embed_pool path right-pads. Fix = monkeypatch `_pad_and_stack` to left-pad + set `ctx.padding_side='left'`
(same as the chunk-tda extractor); left-pad with a correct attention mask gives an identical pooled gene, so
the reused 1x rows stay consistent. New: embed_pool_incremental.py (reuses 1x genes, embeds only new pids),
run_2x_eval.sh (apply 2x adapter per field + eval + #65 retrieve, restores canonical 1x genkron).
