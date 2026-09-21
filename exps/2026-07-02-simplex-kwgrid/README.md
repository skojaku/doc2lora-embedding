# Fusion simplex: abstract decode + copy/compose metrics (#60)

At every barycentric cell of a 3-corner fusion simplex (RES=12 → 91 cells) we fuse the
three corner papers and decode with each channel, then map **how the output blends the
corners** (RGB), **how extractive it is** (copy-rate), and **how proportionally it tracks
the mix** (fusion curve). The three channels:

- **Doc2LoRA** — adapter value-interpolation of the corner genes (Qwen-4B)
- **In-context** — one prompt, proportional blend instruction
- **ICAE** — convex mix of the 128 memory slots (Mistral-7B)

## Pipeline

1. **Corner selection** — `sample_far_triples.py`
   Over-samples random valid triples per PACS-distance stratum (L1 near … L5 far) and
   keeps the far half — mild bias toward separated corners that **preserves the L1→L5
   distance gradient** (a max-min search saturates ≈1.1 and flattens it). Ranks by
   min-pairwise SBERT (`data/aps/embeddings/sbert_allmpnet.npz`) distance, keeps top X of
   FACTOR·X. Writes `corners_strat<NN>_mild.json` (00–19 L1 … 80–99 L5) + `mild_manifest.json`.

2. **Decode** — `decode_absfollow.py` (Doc2LoRA + In-context), `decode_absfollow_icae.py` (ICAE)
   Per cell, two turns in one conversation: (1) generate an **abstract** from the fused
   embedding, (2) ask for exactly **5 keyword phrases**. Both saved. Corner source =
   keyphrases + lead abstract (`SRC_KW=1`). Metrics use the abstract; keywords are kept for
   figure clarity only. Writes `results/absfollow_<set>_kwsrc[_icae].json`.

3. **Metrics + figures** — `absf_metrics.py`
   Corner anchor = the **original lead abstract** embedding (all-mpnet), same modality as the
   decoded abstract (this alone removes the earlier "red-flood" — comparing a full abstract
   to short keyphrase-centroids collapses all corners to cos≈0.7).
   - **RGB** `w = softmax(cos(decoded, source_j)/τ)`, τ=0.30, γ=1.5 (γ = colour sharpen only).
   - **copy-rate** = fraction of abstract word-tokens verbatim in the source leads (cividis).
   - **fusion curve** = measured corner-weight vs ideal barycentric weight (diagonal = smooth
     proportional fusion; flat = confabulated hedge), scalar `r` per method.

## Reproduce

```bash
export PYTHONPATH=$DOC_TO_LORA_SRC
export DOC2LORA_CKPT=data/agent_assets/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin

# 1. corner selection (CPU, ~2 min)
python sample_far_triples.py --per-stratum 20 --factor 2 --seed 0

# 2. decode the two illustrative sets (GPU, ~35 min; 4 GPUs)
SRC_KW=1 KWTAG=_kwsrc CUDA_VISIBLE_DEVICES=0 python decode_absfollow.py      strat00_mild
SRC_KW=1 KWTAG=_kwsrc CUDA_VISIBLE_DEVICES=1 python decode_absfollow.py      strat80_mild
SRC_KW=1 KWTAG=_kwsrc CUDA_VISIBLE_DEVICES=2 python decode_absfollow_icae.py strat00_mild
SRC_KW=1 KWTAG=_kwsrc CUDA_VISIBLE_DEVICES=3 python decode_absfollow_icae.py strat80_mild

# 3. figures  (compute needs 1 GPU for SBERT; --replot is CPU-only from cache)
KWTAG=_kwsrc CUDA_VISIBLE_DEVICES=0 python absf_metrics.py strat00_mild strat80_mild
KWTAG=_kwsrc                        python absf_metrics.py --replot strat00_mild strat80_mild
```

Figures → `figs/absf_rgb.png`, `figs/absf_copyrate.png`, `figs/absf_fusion_curve.png`.
Knobs (`absf_metrics.py`): `TAU=0.30`, `GAMMA=1.5` (RGB); cividis copy-rate.

## Findings (strat00_mild L1 near / strat80_mild L5 far)

| metric | Doc2LoRA | In-context | ICAE |
|---|---|---|---|
| copy-rate near | 0.61 | 0.62 | 0.68 |
| copy-rate far  | 0.54 | **0.47** | **0.71** |
| fusion curve r | 0.89 / 0.90 | 0.72 / **0.64** | 0.89 / 0.91 |

- **Doc2LoRA = smooth proportional compose** — tracks the mix (r≈0.9), low/flat copy-rate
  (paraphrases), balanced RGB gradient ≈ ideal.
- **ICAE = winner-take-all copy** — also tracks (r≈0.9) but by picking the dominant source
  verbatim (copy-rate 0.68→0.71, highest), hard blocky RGB regions.
- **In-context = confabulated blend** — hedges (r≈0.65, never commits), and its copy-rate
  **collapses at far distance** (0.47) — can't stitch 3 disparate sources, muddy RGB.

So **copy-rate** separates compose (Doc2LoRA) from copy (ICAE); the **fusion curve**
separates the two latent methods from in-context's hedging. Pulling the corners apart
(mild sampling) sharpens all three views without saturating the PACS-distance gradient.

## Files

```
sample_far_triples.py       corner selection (mild, gradient-preserving)
decode_absfollow.py         Doc2LoRA + In-context abstract→keyword decode
decode_absfollow_icae.py    ICAE abstract→keyword decode
absf_metrics.py             RGB + copy-rate + fusion-curve figures
simplex_common.py           corner/grid plumbing
workflow/plot/_simplex.py   house simplex drawing (colours, bary_to_xy, rgb_blend)
workflow/plot/_style.py     house matplotlib style
corners_strat*_mild.json    100 corner triples + mild_manifest.json
results/absfollow_*_mild_kwsrc*.json   decoded abstracts+keywords (the 2 illustrative sets)
figs/absf_{rgb,copyrate,fusion_curve}.{png,pdf}
```

External deps (imported by path): `doc2lora_legacy` (package), `simplex_fusion_icae`
(exps/2026-06-12-idea-simplex), `sample_sci_triples` (exps/2026-06-20-simplex-metrics).

## Prompt-sensitivity arm (edge decode)

The paper's §4.3 edge experiment issues ONE instruction at every interpolated point
("Write a detailed abstract (four to six sentences) describing this research topic: its
problem, methods, and findings."). Appendix `app:prompt-sensitivity` does report a
paraphrase sweep, but over a *different* prompt family — the 2–3 sentence "combined
research idea" wording in `workflow/scripts/groupc/prompts.py` — so the two claims §4.3
actually makes had never been tested against paraphrase:

1. Doc2LoRA and ICAE track the requested mixing weight; in-context sits at the midpoint.
2. Copy rate orders ICAE > in-context > Doc2LoRA.

`psens_prompts.py` holds four semantically equivalent wordings of that instruction, index
0 being the manuscript's own. `decode_absfollow.py` and `decode_absfollow_icae.py` select
one with `ABS_PROMPT_ID` (default 0 → byte-identical to the reported run) and skip the
keyword turn with `SKIP_KW=1`. The in-context arm keeps its recipe framing (two documents
+ barycentric percentages) and swaps only the trailing instruction, so all three methods
move along the same paraphrase axis.

```bash
# 20 pairs per stratum, strata L1 (near) and L5 (far), prompts 1..3 (0 already decoded)
snakemake kg_psens_all -j4 --resources gpu=2 \
  --allowed-rules kg_psens_decode_doc2lora kg_psens_decode_icae kg_psens_score kg_psens_all
```

`psens_edge_score.py` re-scores both paper panels per paraphrase with the panels' own
definitions — landing position `(d-A)·(B-A)/|B-A|²` from `pair_axis_metrics.py`, copy rate
from `pair_axis_copyrate.py` — and adds a cross-prompt similarity with its across-point
null. Outputs `results/psens_edge.json`, `figs/prompt_sensitivity_edge.tex`,
`figs/psens_edge_curves.pdf`.

### Result (2026-09-19, 20 pairs x 2 strata x 13 points x 4 prompts = 6,240 decodes)

Both §4.3 readings survive paraphrase; the numbers barely move.

| method | tracking slope (near / far) | copy rate over the edge (near / far) |
|---|---|---|
| Doc2LoRA   | 1.11–1.12 / 1.19–1.20 | 0.50–0.59 / 0.48–0.61 |
| In-context | 0.29–0.36 / 0.37–0.41 | 0.53–0.57 / 0.50–0.57 |
| ICAE       | 1.19–1.20 / 1.23–1.24 | 0.64–0.79 / 0.63–0.82 |

Ranges are across the four prompts. The separation the section rests on — latent
interpolation tracks the requested weight (slope ≈ 1.2) while in-context does not
(≈ 0.35) — is a factor of three under every wording, and no paraphrase moves any slope
by more than 0.04. Pointwise the latent curves are sigmoidal rather than straight: flat
at 0.06–0.10 out to a third of the way along, a sharp rise through the midpoint, then a
plateau above 0.9 (Doc2LoRA far, reported prompt: .06 .06 .06 .07 .10 .24 .53 .81 .92 .93
.94 .94 .94). In-context sits at 0.41–0.55 across the whole interior and moves only at the
two ends. Across prompts the latent curves shift by at most 0.06 at any point; in-context
by at most 0.11.

Copy rate is the weaker claim, and paraphrase is not what weakens it. ICAE copies most
under all four prompts (edge-mean 0.72–0.75), never overlapping the others. Doc2LoRA and
in-context do overlap, and the paper's ordering does not hold:

| edge-mean copy rate | prompt 0 (reported) | 1 | 2 | 3 |
|---|---|---|---|---|
| Doc2LoRA near   | 0.558 | 0.544 | 0.558 | 0.549 |
| in-context near | 0.546 | 0.544 | 0.545 | 0.557 |
| Doc2LoRA far    | 0.561 | 0.549 | 0.567 | 0.557 |
| in-context far  | 0.523 | 0.523 | 0.521 | 0.533 |

Near, the two are indistinguishable and their order flips between wordings. Far, Doc2LoRA
copies *more* than in-context under every wording including the manuscript's own — the
opposite of "\doctolora the least, in-context in between". That is a statement about the
data, not about prompt stability. "ICAE copies the most" is the part that survives.

Cross-prompt similarity of the decoded abstracts is .886 (Doc2LoRA), .924 (in-context),
.923 (ICAE), against across-point nulls of .146 / .256 / .125 — the prompt rewrites the
sentences far less than the coordinate does.

Two things found on the way that are not about paraphrase:

- The copy-rate figures §4.3 quotes (ICAE .63–.65, Doc2LoRA .49–.52, in-context .52–.56)
  do not reproduce from `results/pair_axis_copyrate_pairaxis_{L1,L5}.json`, which gives
  ICAE .62–.78, Doc2LoRA .48–.59, in-context .50–.56 over the edge (and .73 / .56 / .54
  as per-stratum means). The ordering holds; the numbers need re-deriving from whatever
  produced them.
- `exps/2026-07-02-simplex-kwgrid/results` is an untracked symlink into the data store
  (`/data2/.../data/exps/2026-07-02-simplex-kwgrid/results`, 961 files), the same pattern
  the other experiment dirs use. The `paper/iclr2026` branch, uniquely, *tracks* seven
  JSONs at that path, so checking it out replaces the symlink with a real directory
  holding those seven and hides the rest. Nothing is lost — the decodes and
  `psens_edge.json` are intact data-side — but on that branch the analysis scripts read a
  seven-file directory and any rerun writes into the repo instead of the data store. Worth
  either untracking those seven or making the path a symlink everywhere.
