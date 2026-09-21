# Is the doc2lora gene more decodable than raw LLM activations? (issue #152)

The paper claims the \doctolora latent is a **decodable** embedding space. Text encoders
cannot test that claim — they have no decoder. Raw LLM activations can: they are the
same kind of object (a neural representation feeding a generator), they decode with no
training via activation patching, and they come from the very model the gene decodes
through. It is the fairest and most dangerous baseline available.

## Setup

Both arms run on **Qwen3-4B-Instruct-2507** (36 blocks, hidden 2,560) — the base the
\doctolora qwen checkpoint decodes through. Generator, tokenizer and instruction text
are identical across arms; **only the injected object differs**. A second base model was
considered and dropped (2026-09-18): it would reintroduce the generator confound this
design exists to remove.

| arm | object | floats |
|---|---|---|
| `gene` | \doctolora full-rank `norm_lora_emb` (36 × 8 × 512) | 147,456 |
| `act_mean` | layer-`l` hidden states, mean-pooled over non-pad tokens | 2,560 |
| `act_last` | layer-`l` hidden state at the last non-pad token | 2,560 |

Not size-matched, deliberately (`act_k36` dropped 2026-09-18): the activation arm is the
**smaller** object, so a gene win carries that caveat and an activation win would be the
stronger result.

Scoring: **M1** (rapidfuzz token-set vs the official PACS label) and **M3** (nearest
category, five judges). M2 is out of scope (2026-09-18).

## The decoder is the experiment

Patching reads a hidden state from the document's forward pass at block `l` and writes
it into a *separate* decode prompt at a placeholder position, same block `l`. Read and
write use the same interface — the INPUT to block `l`, i.e. `hidden_states[l]` in HF's
indexing (index 0 = embeddings) — via a `forward_pre_hook`, so no representational
offset is introduced. No training, which is what makes it comparable to a gene decode.

The hook fires only on prefill (`seq_len > 1`); firing on decode steps would overwrite
every generated token.

## Two bugs found before any result was trusted

**1. The placeholder index was wrong.** BPE merges the placeholder with its neighbours —
Qwen turns `"? \n\n"` into a single `'Ġ?ĊĊ'` token — so locating it by string offset
pointed at the token `'In'` instead. `build_decode_inputs` now concatenates token ids
(head + slot ids + tail), making the indices exact by construction.

**2. The first hook test was self-deceiving.** Pre-hooks run in registration order, and
the probe was registered *before* the patch hook, so it read the tensor as it was before
patching and reported "the patch does not land". Registering the probe inside the patch
context shows the sentinel reproduced exactly (patched row mean 7.000 vs 0.013).

Magnitude was checked and is **not** a problem: at L8 the injected `act_mean` has norm
46.2 against a prompt-token median of 33.3 — the injected vector is if anything larger
than what the model normally sees there.

## The target prompt decides whether anything is readable

With the mechanism verified, the decode still answered "the document is empty" — which
is a property of the *question*, not of the activation. A direct question hands the model
an escape hatch. Four families swept (`diag_prompts.py`, 3 layers × {1,8} slots):

| family | result |
|---|---|
| `question` — "Document: ?\n\nIn one sentence, what is the topic?" | dead. Answers "it is empty" at every layer. |
| `identity` — Patchscopes' `cat -> cat / 1135 -> 1135 / ? ->` | dead. The model copies the demo tokens; the patched slot reads back as `1135`. The published identity prompt is built for single-**token** representations, not pooled document vectors. |
| `repeat` — "Repeat the following text verbatim: ?" | dead. Degenerate output (`€ħ€€ħ…`, runs of zeros). |
| **`continue`** — "The following is a scientific abstract.\n\n?" | **carries signal.** |

Best observed cell, `continue` / L18 / 8 slots, on *Many-body electronic structure and
Kondo properties of cobalt-porphyrin molecules*:

> "…The electronic structure of the system is governed by the interplay between the
> spin-orbit coupling (SOC) and the spin…"

against a floor (nothing injected) of biomedical boilerplate ("a 12-week resistance
training program on muscle strength"). The second test document (*Electron-γ angular
correlations in 134Ba*) did **not** come back on topic, so the signal is real but weak
and has to be measured, not eyeballed.

Sweeping the prompt family and handing the activation arm the winner is deliberate:
fixing a bad prompt by hand would manufacture a "raw activations do not decode" result
that is really "we asked in a form the method never uses" — the crippled-baseline
failure (survey pattern P2) this experiment exists to avoid.

**The floor matters more here than anywhere else.** `continue` has a strong prior toward
biomedical abstracts, so any arm must be scored against what the prompt alone produces,
never in absolute terms.

## Files

| file | what |
|---|---|
| `act_common.py` | model load, activation extraction (`mean`/`last`), exact-index prompt builder, patching decoder, floor decode |
| `diag_patch.py` | mechanism checks: hook fires, index correct, magnitude in range, slots/norm leverage |
| `diag_prompts.py` | target-prompt family sweep |

## Next

1. `layer_sweep.py` — quantify `continue` across layer × slots × {mean,last} with SBERT
   cosine to the source abstract **minus the floor**, on ~50 documents; pick the
   activation arm's configuration from the winner.
2. `decode_compare.py` — M-A (source identification against 99 same-PACS-subfield
   distractors; hard distractors from the start, since random ones previously
   manufactured a win that vanished) and M-B (SBERT fidelity, ROUGE-L), 500 documents.
3. `arithmetic_compare.py` — 28 PACS node means and 100 midpoint pairs in both
   representations, scored by M1 and M3 with bootstrap s.d.

## LatentQA arm — attempted, NOT reportable

A trained activation decoder is the obvious objection to the patching result ("activations
decode badly *under Patchscopes*; train a reader and they would not"), so LatentQA
(`pan2024latentqa`) was run as the baseline's ceiling. It never passed its validity gate,
so **no number from it is reported**.

Setup: released decoder `aypan17/latentqa_llama-3-8b-instruct` (r=16 LoRA) reading layer 15
of Llama-3-8B-Instruct — via the ungated `NousResearch` mirror, since `meta-llama` is gated
on our token. Layer 15 is LatentQA's own default, the same depth our Qwen sweep picked
independently.

### Four upstream breakages at `latentqa@a2dcb6f` (current HEAD)

1. `dataset_utils` renamed `tokenize` → `lqa_tokenize`; `reading.py` still imports the old
   name, so importing it fails outright.
2. Five module-level tables are keyed by the canonical model name and miss the mirror.
   Aliased generically rather than the two we tripped over first: a missed table would not
   raise, it would silently fall back to a different chat template.
3. `reading.py` indexes a `BatchEncoding` as if it were a tensor, so `generate=True` dies.
4. `main()` computes a loss with `generate=False`, but the README documents that same
   command as producing QA pairs. At the **initial commit `0e4eebf`** it did — `ed70fb8`
   repurposed it. `0e4eebf` is the state the README and the released decoder correspond to.

### The controlled reproduction, and its result

Everything controllable was matched to the authors' setup: their commit `0e4eebf` in a
separate worktree, their pinned `peft==0.10.0` / `transformers==4.46.2` in `.venv-latentqa`
(system torch 2.6 shared), their exact veganism prompt from the README, their own
`QUESTIONS`, their code unmodified. The mirror was cleared as a suspect too — vocab 128256,
id 128010 = `<|reserved_special_token_5|>` (their `PAD_TOKEN_IDS` value), eos 128009 — all
matching what the code expects.

Result: every question returns only the decoder's stock opener,

```
Q: What type of persona will the assistant adopt?      A: Sure, I've analyzed the assistant.
Q: How will the assistant express themselves?          A: Sure, I've analyzed the assistant.
Q: Please describe the persona of the assistant.       A: Sure, I've analyzed the assistant.
```

That string is `BASE_DIALOG`'s prefix turn, which is what the model emits when the written
activations have no effect — consistent with the write not landing, though we did not
isolate where.

### Why it is not reported

The claim we could make from this arm would be "a trained activation decoder also fails",
and it would be **false-looking in our favour**: the experiment's stated goal is to show the
gene is more usable than existing methods, and a broken baseline produces exactly that
appearance. The same gate stopped the T2L arm until `lol_636` went 0/80 → 75/80.

What can be said is narrow and about artifacts, not about the method: *we could not
reproduce LatentQA's released reading demo from its released decoder and code.*

None of #152's conclusions depend on this arm.

Files: `latentqa_arm.py` (the arm, runnable once the write path works), worktree
`$LATENTQA_REPO` (clone of `latentqa` at `0e4eebf`), venv `.venv-latentqa`.
