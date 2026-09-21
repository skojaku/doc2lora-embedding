# Text-to-LoRA as a hypernetwork-adapter baseline (issue #151)

Answers the reviewer objection that sank the base method at ICLR 2026 ("no comparison
with text-to-lora baselines"). The question is not whether T2L is a good hypernetwork —
it is whether *generating adapters* is by itself enough to give a decodable space, or
whether the space has to be **learned**. §1 argues this in prose; this experiment
measures it.

## Where the embedding sits

```
D2L:  doc  --f_enc-->  V (147k, Qwen)  --f_dec (linear)-->  (A, B)  -->  dW  --> generator
                        ^ learned intermediate = our embedding

T2L:  text --gte-large-en-v1.5 (frozen)--> z0 (1024)
           --TaskEncoder: Linear(1024,64)+LayerNorm--> z1 (64)
           --mixer/mlp1-3 + layer&module embeddings--> heads --> (A, B) --> dW --> generator
                        ^ frozen off-the-shelf encoder output; z1 is the ONLY
                          document-dependent learned representation in the pipeline
```

Three candidate embedding positions, all measured (**E3, the true mean of the dW
matrices, is explicitly out of scope** — decision 2026-09-18):

| | space | dim / params | per doc (fp16) |
|---|---|---|---|
| **E0** | gte-large output | 1,024 | 2 KiB |
| **E1** | TaskEncoder output | **64** | 128 B |
| **E2** | generated LoRA factors (A, B) | 3,407,872 | 6.5 MiB |
| — | dense dW, if materialized | 671,088,640 | 1.25 GiB |

Everything downstream of z1 mixes in only layer (32-d) and module (32-d) embeddings,
which do not depend on the document. **The whole document bottleneck in T2L is 64
dimensions, and it is a linear map of a frozen encoder's output.**

## Setup (all verified against the released checkpoint)

- hypernet: `SakanaAI/text-to-lora` → `trained_t2l/mistral_7b_t2l` (`hypermod.pt` is a
  plain `state_dict`, `training_task: sft`, `encoder_type: linear`, `shared_AB_head:
  false`, `hypernet_latent_size: 128`, `head_in_size: 512`)
- base / generator: `mistralai/Mistral-7B-Instruct-v0.2` — **the same base as the ICAE
  arm** (`workflow/config.yaml:icae_base_model`), so decodes are comparable at the
  generator level
- LoRA: `r=8`, `alpha=16`, `use_rslora=true`, targets `q_proj`+`v_proj` on all 32 layers
- document encoder: `Alibaba-NLP/gte-large-en-v1.5`, CLS pooling, fp32

Runs in the **main env** (torch 2.6.0 / transformers 4.51.3): the T2L pin
(`transformers==4.46.2`, `vllm==0.5.4`) is only needed for their training/eval harness.
Added deps: `inflect`, `torchmetrics`, `wandb` (import chain of
`hyper_llm_modulator.utils`). No isolated venv.

```
export PYTHONPATH=$T2L_SRC
```

Repo clone: `$T2L_REPO` (outside the tree, like
`doc-to-lora`). `load_t2l` chdirs there during the load because T2L resolves
`chat_templates/<model>/chat_template.jinja` relative to CWD — we decode with **their**
chat template.

## Input policy (issue Decision, 2026-09-18)

Every method receives the paper's **title + abstract, verbatim**. No task-description
rewrite for T2L, no per-method input tuning. The consequence — T2L runs outside its
training distribution — is stated in the manuscript as an interpretation limit, not
patched experimentally. The claim ceiling is:

> As a document embedding, on the same input every other method gets, T2L's pipeline
> does not yield a decodable space.

Nothing stronger. We did not test T2L at what T2L was built for.

## Averaging convention

Doc2LoRA's head is linear and emits both LoRA factors, so averaging embeddings averages
`A` and `B` separately and the mean **stays rank 8**
(`libs/doc2lora/doc2lora/arithmetic.py:internalize_from_norm_lora_emb` →
`hypernet.head` → `_to_lora_dict`). `t2l_common.mean_lora_sd` does the same for T2L —
that is the structural analogue. `mean(B_i @ A_i)` is a different operation (rank 8N)
for *both* methods and is out of scope.

`dw_cos` compares generated adapters exactly via
`<B1 A1, B2 A2>_F = tr((B1^T B2)(A2 A1^T))`, so the 671M-parameter dense dW is never
materialized. Per-document adapters are never stored either (56k docs × 6.5 MiB =
364 GB); `LoraAccumulator` streams the mean.

## Files

| file | what |
|---|---|
| `t2l_common.py` | loader, E0/E1/E2 extraction, factor-space averaging, exact dW cosine, decode |
| `verify_wiring.py` | proves the generated LoRA actually reaches the model |
| `sanity_degeneracy.py` | (A) do different documents give different adapters (B) do endpoints decode |
| `task_validity.py` | control: does the pipeline reproduce T2L's intended behaviour (adapter on vs off on a task the hypernet was trained on) |

## Results

### Wiring (`verify_wiring.py`) — PASS

128/128 LoRA tensors change on apply; next-token logits move substantially against the
zeroed adapter (mean |Δlogit| ≈ 1.05, argmax flips). The `key overlap: 0` line is
cosmetic — `set_peft_model_state_dict` inserts the adapter name (`.default`) itself.
Document A vs document B also differ (mean |Δlogit| ≈ 0.54), so document identity does
reach the weights.

So any failure to decode is **not** a plumbing failure.

### Non-degeneracy + endpoint decode (`sanity_degeneracy.py`, n=60 abstracts / 12 descriptions)

Pairwise cosine, off-diagonal:

| space | APS abstracts (1,770 pairs) | T2L's own task descriptions (66 pairs) |
|---|---|---|
| **E0** gte-large (1,024) | 0.416 ± 0.074 | 0.591 ± 0.053 |
| **E1** TaskEncoder (64) | 0.274 ± 0.181 | 0.067 ± 0.227 |
| **E2** generated dW | **0.857 ± 0.053** | **0.699 ± 0.129** |

The order **inverts** between input and adapter space: task descriptions are more
similar to each other than our abstracts are in gte space (0.591 > 0.416), yet produce
*more* differentiated adapters (0.699 < 0.857). The hypernetwork spreads apart the
inputs it was trained on and compresses the ones it was not — a quantitative statement
of the distribution shift the manuscript has to declare.

Adapters are **not** degenerate (nothing like ICAE's 0.96–0.99), so a decode failure is
not "all adapters are the same".

Endpoint decode collapses anyway. With the corrected prompt format, 8/8 physics papers
decode to a constant:

```
Transient terahertz conductivity in photoexcited silicon nanocrystals -> "Social Sciences"
Ground-state properties of the periodic Anderson model               -> "Social psychology"
Medium-Term Prediction of Chaos                                      -> "Social Psychology"
First limit on neutrinoless quadruple beta decay of Nd150            -> "Social Sciences"
Dynamics of crater formations in immersed granular materials         -> "Social psychology"
```

and **T2L's own task descriptions decode to unrelated text too** (`arc_easy` -> "The
history of the United States of America"). So the failure to read content back is not
an artifact of our out-of-distribution input — it is the absence of any text
reconstruction objective in T2L's training.

### Pipeline validity (`task_validity.py`) — PASS

| task | in `train_ds_names` | adapter OFF | adapter ON | delta |
|---|---|---|---|---|
| `lol_636` (extract/sort unique letters) | **yes** | 0.000 | **0.938** | **+0.938** |
| `gsm8k` | no (held-out) | 0.463 | 0.163 | −0.300 |

`lol_636` is the check that counts: on a task the hypernetwork was trained on, the
generated adapter takes exact-match from 0/80 to 75/80. The base model answers that
task by emitting Python code; the adapter makes it emit the answer. **The pipeline
reproduces T2L's intended behaviour, so the decode failure above is a property of T2L,
not of our setup.**

The GSM8K row is kept as a caution, not as a finding: gsm8k is **not** among the 479
training tasks, and the adapter's effect there is to suppress chain-of-thought (the LoL
tasks are short-answer), which is precisely what GSM8K needs. Choosing it first was a
task-selection error on our side. Do not cite it as evidence about T2L.

Combined with `verify_wiring.py`, the three boring explanations — wrong wiring, dead
adapters, broken loading — are all excluded.

## Next

1. Label arm — 28 PACS nodes, member sets frozen from
   `exps/2026-06-11-baseline-trees/prep_nodes.py` (no resampling, arms stay paired),
   CAP=2000 members like D2L (not ICAE's K=8, so the baseline is not run weak), scored
   by the existing M1/M2/M3 with bootstrap s.d. (#146 — point estimates are not enough).
2. Fusion arm — reuse the L1 (near) / L5 (far) strata of
   `workflow/rules/simplex_kwgrid.smk`, 50 pairs each, the same 13 points, scored by
   `pair_axis_metrics.py` and `workflow/scripts/fusion_vs_copy.py`.
3. Retrieval row — gte-large in `workflow/scripts/text_encoders.py`, labelled honestly
   as "Text-to-LoRA's native coordinates = gte-large", noting the #142 D6 decision so it
   does not read as a new-encoder addition.
4. `workflow/rules/t2l.smk` (template: `workflow/rules/icae.smk`), and
   `t2l_gte` / `t2l_hidden` / `t2l_dw` in `label_eval_prep.py:METHOD_FILES`.
