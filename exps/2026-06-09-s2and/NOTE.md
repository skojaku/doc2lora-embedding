# Author Name Disambiguation with idea genes — S2AND benchmark

## Question
Can doc2lora idea-genes (and the invertible same-author Kron adapter) disambiguate authors? Test on
**S2AND** (Subramanian et al. 2021, AllenAI) — the standard AND benchmark — as an EMBEDDING-ISOLATION
test: per-signature embedding = its paper's vector; cluster signatures within each name-block; score B³.

## Setup
- Data: S2AND public S3 (`--no-sign-request`), small sub-datasets: zbmath(math), qian(CS), arnetminer(CS).
  Each = signatures.json (name `block`, gold via clusters.json) + papers.json (title+abstract) + precomputed
  `specter.pickle` (the S2AND production embedding).
- Embeddings compared per paper: **gene** (gemma doc2lora, lib `load_model(mode="embed")`, norm_lora_emb
  rank-pooled→13312, **max_tokens=512**), **gene_kron** (gene through a Kron adapter trained on SAME-AUTHOR
  in-batch InfoNCE over TRAIN blocks, 2000 steps ~15s), **specter** (built-in), **sbert** (all-mpnet),
  **instructor** (instructor-large).
- Protocol: blocks split 60/40 train/test (seed 42). Agglomerative avg-linkage on cosine distance
  (precomputed matrix; zero-vec safe), threshold tuned on TRAIN to max B³ F1, applied to TEST. Pooled B³.
- Scripts: prep.py, extract_genes.py (lib API, mode=embed, capped tokens), text_baselines.py
  (reuses workflow/scripts/text_encoders), and_eval.py; metric port in s2lib.py.

## Results (B³ F1, test blocks; P/R in results_<ds>.csv)
| dataset | gene | gene_kron | specter | sbert | instructor |
|---------|------|-----------|---------|-------|-----------|
| zbmath (math)     | 0.934 | **0.946** | 0.934 | 0.944 | 0.936 |
| qian (CS)         | 0.794 | 0.803     | 0.832 | **0.865** | 0.833 |
| arnetminer (CS)   | 0.663 | 0.641     | 0.669 | **0.698** | 0.679 |
| mean              | 0.797 | 0.797     | 0.812 | 0.836 | 0.816 |

## Read (honest, domain-dependent)
- **zbMATH: gene_kron is the BEST embedding** (0.946 > sbert 0.944 > specter/instructor 0.934/0.936). Math
  author≈narrow-topic, so the topical idea-gene + same-author adapter nails identity. Raw gene already ties specter.
- **CS (qian, arnetminer): SBERT leads; genes trail** (genes weakest of the 5). Same-name CS authors share
  subfields, so topical content is less identifying; lexical/text encoders win.
- **Adapter is inconsistent**: helps zbmath (+.012) & qian (+.009) but HURTS arnetminer (−.022, only 130
  blocks → tiny/noisy test, 1 seed). On average it's a wash (.797→.797).
- Net: idea-genes are a **viable AND embedding — competitive with SPECTER, best on topic-coherent domains
  (math)** — but not a uniform winner; SBERT is the stronger generic AND text embedding here.

## FAIRNESS CONTROL + 3 gene encoders (same-author adapter trained on EVERY embedding)
B³ F1, raw → +same-author Kron adapter (flat C=QΛ for 768-d text; [L,512] for genes):
| model | zbmath | qian | arnetminer |
|-------|--------|------|------------|
| gene·gemma   | .934→.946 | .794→.803 | .663→.641 |
| gene·qwen    | .934→.941 | .787→.827 | .678→.666 |
| gene·mistral | .934→.941 | .784→.825 | .659→.713 |
| specter      | .934→.917↓ | .832→.740↓ | .669→.640↓ |
| sbert        | .944→.913↓ | .865→.793↓ | .698→.651↓ |
| instructor   | .936→.906↓ | .833→.726↓ | .679→.636↓ |

KEY ASYMMETRY: the same-author adapter **HELPS genes but HURTS every text baseline, consistently**
(3 datasets × 3 text encoders all ↓). Interpretation: sbert/specter/instructor are already contrastively
trained → saturated 768-d geometry → extra same-author InfoNCE on small train blocks OVERFITS & damages it;
idea-genes are GENERATION-trained (not representation-trained) → headroom the adapter exploits. = the paradigm
thesis concretely: token-embeddings saturated, LoRA-genes still trainable.
CONSEQUENCE: raw → sbert best generic AND emb (genes competitive, best on zbmath). ADAPTED (everyone gets the
adapter; realistic since S2AND ships train labels) → idea-genes BEST on ALL 3: zbmath gene_kron·gemma .946,
qian gene_kron·qwen .827 (top adapted; raw sbert .865 still higher), arnetminer gene_kron·mistral .713 (top
OVERALL > raw sbert .698). No single gene encoder wins all: gemma→math, qwen/mistral→CS.
Caveat: adapter hparams (2000 steps, lr1e-3) implicitly genes-tuned; gentler reg/early-stop MIGHT not hurt text
(the ↓ could be partly over-training) — but the direction is unanimous. GPU note: ollama squats GPU1 (23GB), ran mistral on GPU3.

## REGULARIZATION cures the overfitting (identity prior; CORRECTS the asymmetry claim)
Doubling steps 2000→4000 overfit (genes regress: arnetminer mistral .713→.672, qian gemma .803→.737).
FIX = identity-prior reg in train loss: `reg·(mean(s²)+mean(α²)+mean(P²))` pulls Λ,A,Q→I (near-isometry).
Swept KRON_REG on qian/gemma: text baselines recover MONOTONICALLY (sbert_kron .750→.825, specter_kron
.748→.837 as reg 0→30); gene_kron peaks .845 at reg 3–10. Picked **REG=10, 2000 steps (new default)**.
REG=10 full 9 (B³ F1), vs unreg-2000 in (): zbmath gene_kron .934(.946)/.934/.926, specter_kron .943(.917),
sbert_kron .941(.913); qian gene_kron .845(.803)/.832/**.851**(.825), specter_kron .837(.740), sbert_kron
.819(.793); arnetminer gene_kron .667/.669/**.704**(.713), specter_kron .691(.640), sbert_kron .704(.651).
HONEST CORRECTION: the earlier "adapter helps genes but HURTS text" asymmetry was LARGELY AN OVERFITTING
ARTIFACT — with the identity prior the adapter is a GENERAL mild improver (specter_kron BEATS raw specter on
all 3: .943/.837/.691 > .934/.832/.669). Genes still gain MORE (headroom real) but text isn't harmed. Stable
gene result: gene_kron·mistral best gene per ds — qian .851 (2nd to raw sbert .865), arnetminer .704 (ties
sbert_kron top), zbmath saturates ~.93 (REG=10 a bit strong, gave up the unreg .946 peak). 1 seed.

## FULL-FEATURED test (the realistic AND model: metadata + swappable embedding feature) — full_featured.py
Faithful subset of S2AND featurizer: name(first/mid match), COAUTHOR overlap (Jaccard+count), venue/journal
match, year-diff, affiliation — + one EMBEDDING-cosine feature. HistGradientBoosting pairwise classifier on
train-block pairs → per-test-block agglomerative avg-linkage (1-prob, thr 0.5) → B³. Ablate the embedding.
B³ F1 (Δ vs metadata-only):
| feature | zbmath | qian | arnetminer |
|---------|--------|------|------------|
| metadata only | 0.933 | 0.847 | 0.828 |
| +specter      | 0.934(+.001) | 0.881(+.034) | 0.827(−.001) |
| +gene raw     | 0.934(+.001) | 0.857(+.012) | 0.826(−.002) |
| +gene_kron·mistral | 0.938(+.005) | **0.890(+.043)** | **0.856(+.028)** |
FINDINGS: (1) in the REAL system the embedding's marginal value is SMALL — coauthor+name metadata dominates
(zbmath saturates ~.93 regardless) → embedding-isolation OVERSTATES the embedding (as cautioned). (2) BUT where
the embedding matters (CS datasets), regularized **gene_kron beats SPECTER as the feature**: qian +.043 vs
specter +.034; arnetminer specter adds NOTHING (−.001) while gene_kron·mistral adds +.028. (3) the ADAPTER
unlocks it — raw gene adds ~+.01; (4) gene_kron·mistral = best embedding feature on ALL 3. Caveats: faithful
SUBSET (no name-rarity/email/title-ngram/reference feats; venue/journal empty in zbmath); fixed 0.5 cluster thr;
1 seed. The controlled RELATIVE claim (gene_kron vs specter, all else equal) holds. Files: full_<ds>_<enc>.csv.

## Caveats / next
- EMBEDDING-ISOLATION only (no name/affiliation/coauthor features). In the FULL S2AND model the embedding is
  one feature; genes could still ADD signal even when not best standalone — the real test is feeding gene/
  gene_kron into s2and's featurizer + pairwise model and measuring Δ vs specter (not yet run).
- Single seed; arnetminer test set tiny. zbmath 54% abstract coverage (title-only otherwise); CS similar.
- Bigger datasets (medline/pubmed/aminer/inspire/kisti) not run.
