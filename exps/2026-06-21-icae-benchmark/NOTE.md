# ICAE benchmark + invertible per-token adapter (issue: ICAE parity with doc2lora)

**Question.** How does ICAE (In-Context Autoencoder, Mistral-7B, 128 memory slots) compare to doc2lora
genes and the text baselines on the *same* benchmark suite — and can we train an ICAE adapter the way
we train doc2lora's general kron adapter, kept *invertible per token* so the compressed memory stays
decodable (the ICLR "task-competitive AND invertible/decodable" property)?

## Design

ICAE is **one encoder-agnostic method**: compress a document (≤512 tokens → 1 segment) into 128 memory
slots of dim 4096; the benchmark vector = **mean over the 128 slots → 4096-dim** (the standard ICAE
embedding). `4096 = 8×512`, so it also plugs into the existing `kron.KronAdapter`.

**Two ICAE variants, mirroring doc2lora's `genes` and `genkron`:**
- `icae`          — raw pooled embedding (a text baseline, like sbert/specter/instructor).
- `icae_genkron`  — the **invertible per-token adapter** applied to the slots, then pooled.

**The adapter** = `KronAdapter(L=128, d=4096)`: per memory token a **common** invertible map
`C = Q·diag(exp(s))` (shared rotation + dimensional scaling) plus a **per-token** scalar `a_l=exp(alpha_l)`.
Exactly the design the user specified. Trained on the **same OpenAlex citation triplets** as doc2lora's
general adapter (42k triplets over a 167k pool), SPECTER2-style InfoNCE with hard+easy negatives;
loss/embedding = mean over the 128 adapted slots. Because `C` is orthogonal×positive-diagonal and
`a_l>0`, the slot map is exactly invertible → adapted slots invert back to the original slots and decode
through the frozen ICAE decoder (`decode_roundtrip.py`).

**S2AND** also gets a same-author `icae_kron` for free (the fairness control trains the same adapter on
every embedding).

## Minimal-subset trick (the costly part)

Generating ICAE embeddings for the **whole** field corpora is unrealistic (economics 565k, psychology
987k, aps 644k genes). `collect_field_ids.py` computes the **exact paper-id union each benchmark
touches** and we embed only that:

| field | full common | topic(fixed 60k) | np (cohorts+fut-pool) | collab | **union embedded** |
|-------|------------:|-----------------:|----------------------:|-------:|-------------------:|
| economics  | 565,475 | 60,000 | 231,543 | 69,057 | **268,276** |
| psychology | 986,800 | 60,000 | 406,164 | 102,581 | **442,058** |
| aps        | 410,313 | 60,000 | 218,631 | 10,038 | **246,487** |

The next-paper negative pool dominates. The **topic subsample is fixed** (`{field}_topic_ids.parquet`,
read by `eval_all.py` via `TOPIC_IDS_FILE`) so every method scores the identical 60k regardless of
coverage — otherwise adding ICAE's partial coverage would shrink the shared `common` set for everyone.
S2AND datasets are small (2.8k–36k papers) → embed in full.

Only the **167k OpenAlex pool slots** are persisted (`pool_slots.npy`, ~175GB fp16, transient — needed
for adapter training). Benchmark papers compress → pool → store 4096-dim only (no slot storage).

## Files
- `icae_lib.py`         — `load_icae`, reference `compress_one`, batched left-padded `compress_batch` (cos>0.999 vs reference).
- `collect_field_ids.py`— minimal touched-id union + fixed topic subsample per field.
- `embed_pool_slots.py` — pool → `pool_slots.npy` [167k,128,4096] fp16.
- `train_icae_adapter.py` — invertible per-token adapter → `adapter_icae.pt`.
- `embed_apply.py`      — embed a benchmark subset → `icae{,_genkron}` npz (raw + adapted, pooled).
- `decode_roundtrip.py` — invertibility/decodability demo.
- `test_smoke.py`       — batched-vs-single + identity-at-init + invertibility (all pass).

## Run
`snakemake icae_all --rerun-triggers mtime -j2`  (rules in `workflow/rules/icae.smk`).
Eval reuses `kron-adapter/eval_all.py` (`INCLUDE_ICAE=1`) and `s2and/and_eval.py` (`OUT_SUFFIX=_icae`),
compared against qwen genes/genkron + sbert/specter2/instructor.

## Status / results — DONE
Smoke PASS (batched compress cos 0.9999; adapter identity-at-init cos 1.0; invert roundtrip cos 1.0).
Adapter trained on 42k OpenAlex triplets (ema 6.2->0.94). Full table in RESULTS.md.

**Headline: raw ICAE is a weak embedding; the invertible per-token adapter (`icae_genkron`) rescues it
to doc2lora-genkron / text-baseline tier.** Same adapter recipe (master-OpenAlex citation triplets)
that helps doc2lora genes helps ICAE slots.

Next-paper AUC (raw icae -> icae_genkron): eco 0.672->0.912, psych 0.662->0.911, aps 0.724->0.952.
Collab AUC: eco 0.561->0.625, psych 0.583->0.666, aps 0.773->0.847 (beats both qwen variants).
S2AND B3 (raw->genkron): pubmed 0.274->0.823, kisti 0.568->0.773, qian 0.732->0.853 (>gene_genkron),
arnetminer 0.577->0.694, zbmath ~0.93 (saturated). doc2lora's edge is decode/interpretability, NOT
raw benchmark numbers — ICAE ~= doc2lora once both carry the citation adapter.

## Bench subset (issue: shrink the benchmark workflow)
Generalized the subset trick to doc2lora + all baselines: `make_subset.py` slices existing embedding
npz (no recompute) to the eval-touched union -> `data/bench/<field>/embeddings` (+ s2and copy). Eval
reads `BENCH_ROOT` (eval_all, score_pool_field) / `BENCH_S2_PROC` (and_eval). Next-paper cohorts pinned
(`NP_COHORT_FILE`) so bench == full is coverage-independent. PROVEN identical: byte-identical rows
(verify_subset) + collab/np AUC identical to 8 digits (verify_eval_cpu). Rules in `workflow/rules/bench.smk`.
Full APS (data/aps) kept separate (other experiment). arxiv + eco/psych full field embeddings removed
(~713G freed) after byte-verify; per-field-kron + layer-bands (the only full-genes consumers) are done.
