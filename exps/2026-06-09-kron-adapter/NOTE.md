# Kronecker citation adapter: making frozen doc2lora genes task-competitive while staying invertible

## Question
doc2lora genes are *generation*-trained and frozen — we cannot retrain the hypernetwork for a task.
Can a **lightweight, invertible, post-hoc** adapter on top of the frozen gene make the representation
**at least as effective as token-as-embedding** (SBERT/SPECTER2/INSTRUCTOR), *without* sacrificing
decodability? This is the missing pillar for the "LoRA-as-embedding" paradigm claim.

## Method
Gene per paper = `G ∈ R^{L×512}` (L=26 gemma layers, mean-pooled over rank/query). Transform:
```
Z[l,:] = a_l · (G[l,:] @ C),   C = Q·diag(exp s),  a_l = exp(alpha_l)
A = diag(exp alpha)  (L params, per-LAYER scale)   [decided: diagonal, not full 26×26]
B = I                (query axis already pooled in the npz)
C = Q·Lambda         (512×512: Q orthogonal via matrix_exp of skew; Lambda=exp(s) positive diag)
```
Exactly invertible factor-wise (`W⁻¹ = A⁻¹⊗C⁻¹`), identity at init. **262,682 params** (vs dense
13312² = 177M, 675×). Induced metric `WᵀW = (AᵀA)⊗(CᵀC)`; Q chooses directions, Λ/A reshape geometry.

**Training** (`train_apply.py`): SPECTER2-style citation contrastive. In-batch InfoNCE over 3M
citation edges (economics `citation_net`, restricted to gene-covered papers), τ=0.05, + simple
log-det volume reg `β(Σs)²+(Σα)²`, β=1e-3. 4000 steps, bs=256, **70 s on one GPU** (genes frozen,
no LM in the loop). Then apply to all 565k genes → `gemma_kron_emb.npz` (same format as vanilla).

**Eval** (`eval_all.py`), economics, 5 representations, 3 tasks (reuses `eval_collab2.author_cos`,
`nextpaper_layerbands`, + cosine-kNN topic on `paper_topics.main_class`, 22 classes, 60k papers 80/20):

## Results (economics, gemma base, 1 seed)
| model       | collab AUC | np AUC | topic F1 | topic acc |
|-------------|-----------|--------|----------|-----------|
| gemma (vanilla) | 0.589 | 0.793 | 0.221 | 0.625 |
| **gemma_kron**  | **0.642** | **0.931** | **0.343** | **0.720** |
| sbert       | 0.653 | 0.939 | 0.386 | 0.737 |
| specter2    | 0.629 | 0.905 | 0.317 | 0.708 |
| instructor  | 0.617 | 0.878 | 0.356 | 0.721 |

- **The adapter helps on all three**: collab +0.053, **next-paper +0.138**, topic F1 +0.122 / acc +0.095.
  Takes vanilla doc2lora from clearly-worst to competitive.
- **Beats SPECTER2 and INSTRUCTOR** on both temporal author tasks (collab, next-paper); on topic it
  beats SPECTER2, trails INSTRUCTOR slightly.
- **Trails SBERT on all three but by small margins** (collab .642 vs .653; np .931 vs .939; topic
  .343 vs .386) — vs a large gap for vanilla. "Competitive, not dominant."

## THREE-FIELD benchmark (economics, aps, psychology; arxiv excluded — no citation net)
NEXT-PAPER (headline): gene+kron WIN/TIE sbert in all 3 fields — econ qwen_kron .941 (sbert .939), aps
mistral_kron .970 (.954), psych qwen_kron .944 (.937). Robust across fields & encoders.
PSYCHOLOGY (cleanest "matches sbert on ALL tasks"): qwen .606→qwen_kron .692 collab / .799→.944 np /
.207→.364 topic — vs sbert .692/.937/.364 ⇒ collab TIE, np BEAT, topic TIE. gemma_kron .682/.937/.347.
COLLAB is FIELD-DEPENDENT: adapter HELPS econ (qwen +.059) & psych (gemma/qwen +.086 each) but HURTS aps
(all encoders) — citation-geometry vs who-coauthors alignment varies by field, NOT a gene flaw (text moves
same way per field). TOPIC: genes tie/beat sbert on psych(.364=.364)/aps(.596>.573), below on econ(.352<.386).
Files: results_psychology_{gemma,qwen}.csv, eval_{gemma,qwen}_psych.log.

## Multi-encoder (8k steps = converged; gemma 8k == 4k, training-length is a non-issue)
| field | enc | collab base→kron | next-paper base→kron | topic-F1 base→kron | sbert col/np/top |
|-------|-----|------------------|----------------------|--------------------|------------------|
| econ  | gemma  | .589→.642 | .793→.929 | .221→.341 | .653/.939/.386 |
| econ  | qwen   | .593→.652 | .814→.941 | .249→.352 | .653/.939/.386 |
| aps   | mistral| .819→.803 | .856→.970 | .593→.596 | .856/.954/.573 |
- **Next-paper = robust large win on ALL 3 encoders** (+.136/.127/.114); kron BEATS sbert for qwen(.941>.939)
  & mistral(.970>.954). The headline, encoder-agnostic.
- **Topic**: helps econ (gemma/qwen), flat on aps-mistral (already high); mistral_kron .596 > sbert .573.
- **Collab**: helps econ (qwen_kron .652 TIES sbert .653) but **slightly HURTS aps-mistral** (.819→.803,
  < sbert .856) — the one negative.
- Mechanism: the adapter is only as good as its supervision — **citation triplets buy citation-aligned
  geometry**, which is what np (within-author topical continuity) & topic reward → uniform wins there.
  Collab-who-coauthors on APS is the task least aligned with citation similarity (sbert dominates it), so
  reshaping toward citations trades a little of it away. Single seed; mistral only on APS (econ has no mistral).
- Files: results_{economics_gemma,economics_qwen,aps_mistral}.csv; train_{gemma,qwen,mistral}8k.log.

## FAIRNESS CONTROL: put the SAME citation adapter on the text baselines (economics)
Flat adapter C=QΛ (d×d, A/B drop out) on sbert/specter2/instructor vecs; train_text_kron.py + eval_text_kron.py.
TEXT raw→+kron: sbert collab .653→.656 / np .939→.940 / topic .409→**.367(↓)**; specter2 .629→.648 / .905→.937 /
.336→.362; instructor .617→.645 / .878→.930 / .360→.373. So the adapter is a GENERAL cheap invertible improver:
big lift for specter2(+.032 np)/instructor(+.052 np), barely moves sbert (near-ceiling) & HURTS sbert topic
(single-objective trades topic for citation-alignment). specter2 still gains despite being citation-trained →
in-domain citation adaptation specializes its general-citation geometry.

THE FAIR FIGHT (all adapted, economics): np — qwen_kron .941 (BEST) > sbert_kron .940 > specter2_kron .937 >
instructor_kron .930 > gemma_kron .929. collab — sbert_kron .656 > qwen_kron .652 > specter2_kron .648. topic —
raw sbert .409 ≫ all adapted .34–.37. VERDICT: gene advantage on next-paper SURVIVES the fair comparison but
shrinks to a hair (qwen_kron tops np); genes within ~.01 on collab; TEXT keeps a real topic edge. So the claim
is "genes MATCH token-as-embedding (marginally best on citation-aligned np) AND uniquely decode" — not dominance.
Kills the "unfair head-start" rebuttal. Files: results_text_kron_economics.csv, train_{sbert,specter2,instructor}.log.

### Fairness control on APS (text adapted too) — confirms the two-field story
TEXT raw→+kron (aps): sbert collab .856→.846 / np .954→.965 / topic .573→.580; specter2 .823→.807 / .936→.965 /
.599→.597; instructor .837→.830 / .900→.955 / .572→.575. FAIR FIGHT (all adapted): np — **mistral_kron .970 BEST**
> sbert_kron .965 = specter2_kron .965 > instructor_kron .955. collab — raw sbert .856 best; **+kron HURTS collab
for EVERY encoder** (sbert→.846, specter2→.807, instructor→.830, mistral→.803) ⇒ the mistral collab dip is NOT
gene-specific, it's a citation-objective vs collab-task MISMATCH (exonerates genes). topic — specter2 .599 ≈
mistral_kron .596 (genes competitive). TWO-FIELD VERDICT: genes+kron are the BEST representation for NEXT-PAPER
on both econ (qwen_kron .941) and aps (mistral_kron .970) even under the fairest comparison; competitive on topic;
collab is the wrong objective for a citation adapter (fix = collab/multi-task supervision, same invertible machinery).

## Mechanism (ties to layer-bands)
Learned per-layer scale `A=exp(alpha)` peaks at **layer 6 (7.84×)**, with layers 0/2/5 boosted and
most deep layers damped to ~0.5–0.8× (ratio 16×). From **citation supervision alone** it rediscovered
the shallow-layer concentration that `2026-06-07-layer-bands` found by hand (gemma best np layer = 6).
Hidden `Lambda` anisotropic (0.16–3.50, cond 22) → it also reshapes the 512-channel geometry, not just
reweights layers. Both factors well-conditioned → invertibility intact (decode path `T⁻¹` cheap).

## Decodability survives adaptation (`decodability.py`, gemma_demo ckpt)
Applied T to the FULL `[26,1,8,512]` norm_lora_emb (C on latent, A on layer, identity on rank — linear,
so consistent with the pooled training), decoded via `internalize_from_norm_lora_emb`.
- **Exactly invertible**: round-trip `cos(E, T⁻¹T E) = 1.000000` for all 4 test papers.
- **Decode preserved**: `decode(E)` and `decode(T⁻¹T E)` are **character-identical** (e.g. solar/wind →
  "development of alternative energy sources in the former Soviet Union"; monetary policy; Nash
  implementation — all faithful and unchanged by the adapter).
- **Arbitrary-point decode still works**: fusion midpoints constructed *in the trained cosine geometry*
  then inverted (`renorm(T⁻¹ renorm(½(T̂Ea+T̂Eb)))`) decode to coherent text ("underdevelopment of
  alternative energy…", "a sufficient alternative to the two-agent model in economic engineering").
- Pre-existing caveat (not the adapter): the gemma_demo decoder is itself imperfectly faithful on some
  inputs (a Spanish product-lifecycle title → "14C-toxaphene in rat liver"); round-trip is identical, so
  we preserve exactly whatever decode fidelity the base checkpoint has.
**Conclusion: the invertible adapter buys task-competitiveness with ZERO loss of decodability — the
paradigm's central promise.**

## Caveats / next
- Single field, single seed, gemma only. np jump is large because vanilla used the diluting all-layer
  flatten; the adapter recovers the shallow signal (sensible, not a bug).
- In-domain (train citations + eval downstream on same corpus, à la SPECTER2). No label leak (collab =
  future windows; topic labels unseen), but citation/topic structure is correlated — report as transfer.
- TODO: (1) decodability-after-adaptation check — decode `T⁻¹(centroid/interp)` vs vanilla, confirm the
  superpower survives; (2) ablate A∈{I,diag,full}, C∈{I,Λ,QΛ}, the "price of invertibility" vs an
  unconstrained low-rank projection; (3) 2nd field + seeds; (4) qwen.

## Files
- `kron.py` (model + InfoNCE), `train_apply.py` (train+dump npz), `eval_all.py` (3 tasks × 5 enc)
- `adapter_economics_gemma.pt`, `results_economics.csv`, `train.log`, `eval.log`
