# Doc2LoRA embeddings ("idea genes") — reproduction workflow

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22842861.svg)](https://doi.org/10.5281/zenodo.22842861)
· [Paper (arXiv:2609.38374)](https://arxiv.org/abs/2609.38374)
· [Project page](https://skojaku.github.io/doc2lora-embedding/)

The code behind *"Doc2LoRA Provides Decodable Representations of Scientific Ideas"*:
the Snakemake workflow that produces every number the paper reports, the scripts it
calls, and the two libraries they import. Code and documentation only — the
manuscript lives elsewhere, and everything the workflow writes is regenerated, not
committed.

A **Doc2LoRA idea gene** (the paper says *Doc2LoRA embedding*; the code says *gene*) is
what you get when a hypernetwork (a network whose output is the weights of another
network) reads a paper and emits a small LoRA adapter for it (a low-rank weight update
that changes what a language model writes). The adapter's latent tensor is the paper's
embedding. Unlike an ordinary sentence embedding, it can be loaded back into the
language model, so any point in the space — including an average or interpolation of
two papers — can be *decoded* into text.

**Every rule here earns its place.** A chain whose result did not reach the paper is
not in this repository.

## Start here: run the workflow on the sample corpus

```bash
./install.sh
cp workflow/config.template.yaml workflow/config.yaml
snakemake sample_check_judged -j4
```

Twenty seconds, CPU only, no downloads, no API key, no licensed corpus. It builds a
small synthetic field, runs the **real** scoring scripts over it (collaboration
prediction, next-paper prediction, topic classification, bootstrap, label metrics,
judge panel), and checks the result against what was planted:

```
PASS  next-paper ranks the methods in the planted order
PASS  topic classification is not saturated
PASS  the round robin ranks the label qualities as planted
PASS  the control ties the exact label with the ground truth
all 18 checks passed                                    (abridged)
```

`make_sample.py` fixes the ordering of the methods before scoring; `check_sample.py`
recovers it from the workflow's outputs, so a pipeline that drops a method or unpairs
the score columns fails. This is a **wiring test, not a result**: it says nothing about
doc2lora, since gene extraction and decoding need a checkpoint. The judge step runs
offline against `local/fuzzy`, a string-similarity stand-in. Use real models with:

```bash
snakemake sample_judge -j1 \
  --config sample_judge_panel="mistral-medium=mistralai/mistral-medium-3.1"
```

## The real thing

| # | You want to | Command | Needs |
|---|---|---|---|
| 1 | Rebuild every reported **table and figure** from archived scores | `python scripts/fetch_artifacts.py results` then `snakemake paper_assets -j4 --rerun-triggers mtime` | CPU, ~224 MB download |
| 2 | Re-score the **name-disambiguation** rows from the vectors | `python scripts/fetch_artifacts.py results s2and` then `snakemake s2and uncertainty -j4` | CPU, +12 GB download |
| 3 | Re-derive **everything from the corpora**, genes included | `snakemake paper_assets -j4` | 1–4 GPUs (≥48 GB total), ~200 GB disk, licensed APS corpus, plus `triplets_1x.parquet` from the `results` tier |

Archived intermediates (per-unit score pools, adapters, benchmark embeddings) are on
Zenodo ([10.5281/zenodo.22842861](https://doi.org/10.5281/zenodo.22842861)), split into
**tiers** named for what each lets you skip: `results` (every GPU- and judge-produced
input of the tables) and `s2and` (the name-disambiguation vectors). The fetch script
unpacks into the repository and checks every file against `data/ARTIFACTS.tsv` by
SHA-256 (`--verify` re-checks what is on disk). Path 3 still needs one file from the
`results` tier, the citation sample the reported transform was trained on.

The 644k-paper APS gene matrices (~137 GB across three encoders) are **not**
distributed: they exceed a Zenodo record and are a deterministic function of the corpus
plus the published checkpoints. Rebuild them with `snakemake all_embeddings`.

`snakemake -n paper_assets` dry-runs the whole DAG (330 jobs from a cold start).
**`REPRODUCE.md`** is the result → code map: for every figure, table, and quoted
number, which rule produces it, plus the tier table and per-chain destinations.

## Setup

`Snakefile` reads `workflow/config.yaml`, which is gitignored so your machine's paths
stay out of git. Copy the template and edit every `<-- set me` line. Eight keys are
read with a hard `config[...]` and must exist: `data_dir`, `aps_paper_table`,
`openalex_paper_table`, `openalex_abstracts`, `mistral_checkpoint_path`,
`qwen_checkpoint_path`, `shard_size`, `gpu_ids`. The sample corpus needs none set.

`./install.sh` builds the conda env `doc2lora` and installs `libs/` editable:
`libs/doc2lora` is the current embed / arithmetic / decode API, `libs/legacy`
(`doc2lora_legacy`) the frozen API the older chains import. Beyond that:

- `ctx_to_lora` (the hypernetwork) is a **separate project**: clone
  [SakanaAI/doc-to-lora](https://github.com/SakanaAI/doc-to-lora) and set
  `workflow/config.yaml:doc_to_lora_src` to its `src/`. Without it the CPU rules still
  run; gene extraction and decoding do not. `hyper_llm_modulator` (Text-to-LoRA) works
  the same way via `t2l_src`.
- `flash-attn` is only for the batched Qwen extractor:
  `pip install flash-attn==2.7.4.post1 --no-build-isolation`.
- EmbeddingGemma (needs `transformers>=4.56`) and `vec2text` each get their own venv
  (`rule setup_emgemma_venv`; `workflow/config.yaml:vec2text_python`).

Secrets go in `.env` (gitignored): `HF_TOKEN` for gated base models,
`OPENROUTER_API_KEY` for the judge rules.

## Data you must obtain yourself

| What | Where | Why it is not here |
|---|---|---|
| APS corpus (644k papers) | request from [APS](https://journals.aps.org/datasets) | licensed; no redistribution |
| OpenAlex snapshot tables | [openalex.org/data-dump](https://openalex.org/data-dump) | ~100 GB upstream |
| S2AND benchmark | [AllenAI S2AND](https://github.com/allenai/S2AND); `rule s2and_download` fetches it | upstream distribution |
| Doc2LoRA checkpoints (Gemma-2-2B, Mistral-7B, Qwen3-4B) | [SakanaAI/doc-to-lora](https://huggingface.co/SakanaAI/doc-to-lora); place under `data/agent_assets/` | model weights |
| Baseline weights: [ICAE](https://github.com/getao/icae), [Text-to-LoRA](https://github.com/SakanaAI/text-to-lora) (`hypermod.pt`), [vec2text](https://github.com/vec2text/vec2text) GTR corrector | upstream projects | third-party weights |

## Layout

```
Snakefile                 24 rule files, one target per reported asset
workflow/rules/*.smk      one file per chain; its header says which result it makes
workflow/scripts/         every script the rules call (bench_data.py: where data lives;
                          text_encoders.py: one build_* per text baseline)
workflow/plot/            figure and table generators
libs/doc2lora, libs/legacy  the two importable APIs
scripts/                  artifact bundling, upload, and fetching
data/                     everything the workflow writes (gitignored); ARTIFACTS.tsv = Zenodo manifest
```

## What is compared against what

Text encoders (SBERT, SPECTER2, INSTRUCTOR, EmbeddingGemma, gte-large), methods you can
read back (ICAE, vec2text, T2L, ActPatch), and an LLM reading the documents directly
(in-context, KeyLLM, BERTopic). The citation transform `g_theta` is an invertible,
Kronecker-factored linear map trained on citing/cited pairs and applied on top of frozen
genes. `REPRODUCE.md` lists every method and the rule file that runs it.

## Citation

```bibtex
@misc{mansuri2026doc2lora,
  title         = {Doc2LoRA Provides Decodable Representations of Scientific Ideas},
  author        = {Mansuri, Chand Sahil and Zachariah, Joel and Kojaku, Sadamori},
  year          = {2026},
  eprint        = {2609.38374},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.38374}
}
```

Mansuri and Zachariah contributed equally. `CITATION.cff` describes the code itself.
