# Doc2LoRA idea genes — reproduction workflow

The code behind *"Doc2LoRA Provides Decodable Representations of Scientific Ideas"*:
the Snakemake workflow that produces every number the paper reports, the scripts it
calls, and the two libraries they import. Code and documentation only — the
manuscript lives elsewhere, and everything the workflow writes is regenerated, not
committed.

A **Doc2LoRA idea gene** is what you get when a hypernetwork reads a paper and emits a
small LoRA adapter for it: the adapter's latent tensor becomes the paper's embedding.
Unlike an ordinary sentence embedding, that vector can be loaded back into the language
model, so any point in the space — including a point you constructed by averaging or
interpolating — can be *decoded* back into text. This workflow measures how well those
vectors work as embeddings and what their decodes look like.

**Every rule here earns its place.** A chain whose result did not reach the paper is
not in this repository, and a rule that no reported result depends on is not in the
workflow.

---

## Start here: run the workflow on the sample corpus

```bash
./install.sh
cp workflow/config.template.yaml workflow/config.yaml
snakemake sample_check_judged -j4
```

Twenty seconds, CPU only, no downloads, no API key, no licensed corpus. It builds a
small synthetic field, runs the **real** scoring scripts over it — collaboration
prediction, next-paper prediction, topic classification, the bootstrap, the label
metrics, and the judge panel — and checks the result against what was planted:

```
PASS  next-paper ranks the methods in the planted order
        qwen_genkron=0.889 sbert=0.877 qwen_kron=0.875 gte=0.853 ... qwen=0.660
PASS  topic classification is not saturated
        spread 0.515 between best and worst
PASS  the round robin ranks the label qualities as planted
        doc2lora=0.917 incontext=0.917 icae=0.724 keyllm=0.581 vec2text=0.205 ...
PASS  the control ties the exact label with the ground truth
all 18 checks passed
```

`make_sample.py` gives every method its own noise level, so the ordering of the methods
is fixed before any scoring happens; `check_sample.py` recovers that ordering from the
workflow's own outputs. A pipeline that drops a method, unpairs the score columns, or
reads a different file than it scores moves the ordering and fails.

This is a **wiring test, not a result**. It says the reporting half of the pipeline is
correct; it says nothing about doc2lora. Gene extraction and decoding need a checkpoint
and are the part a synthetic corpus cannot stand in for.

The judge step runs offline by default against `local/fuzzy`, a string-similarity
stand-in that exercises the panel's plumbing — both presentation orders, the tie rule,
the cache — without spending on inference. Point it at the real thing with:

```bash
snakemake sample_judge -j1 \
  --config sample_judge_panel="mistral-medium=mistralai/mistral-medium-3.1"
```

---

## The real thing

| # | You want to | Command | Needs |
|---|---|---|---|
| 1 | Rebuild every reported **table and figure** from archived scores | `python scripts/fetch_artifacts.py results` then `snakemake paper_assets -j4 --rerun-triggers mtime` | CPU, ~224 MB download |
| 2 | Re-score the **name-disambiguation** rows from the vectors | `python scripts/fetch_artifacts.py results s2and` then `snakemake s2and uncertainty -j4` | CPU, +12 GB download |
| 3 | Re-derive **everything from the corpora**, genes included | `snakemake paper_assets -j4` | 1–4 GPUs (≥48 GB total), ~200 GB disk, licensed APS corpus |

```bash
snakemake -n paper_assets      # the whole DAG, nothing run: 330 jobs from a cold start
```

`REPRODUCE.md` is the result → code map: for every figure, table, and quoted number,
which rule produces it.

## Step 0 — configuration (mandatory)

`Snakefile` reads `workflow/config.yaml`, which is gitignored so your machine's paths
never land in git. Copy the template and edit it:

```bash
cp workflow/config.template.yaml workflow/config.yaml
$EDITOR workflow/config.yaml     # every "<-- set me" line
```

Nine keys are read with a hard `config[...]` and must exist or Snakemake fails while
parsing: `data_dir`, `aps_paper_table`, `openalex_paper_table`, `openalex_abstracts`,
`mistral_checkpoint_path`, `qwen_checkpoint_path`, `icae_weights`, `shard_size`,
`gpu_ids`. Everything else has a default, and the template's values are the ones the
paper used. The sample corpus needs none of them.

## Environment

```bash
./install.sh          # conda env "doc2lora" + editable installs of libs/
```

- `libs/doc2lora` — the current embed / arithmetic / decode API.
- `libs/legacy` (`doc2lora_legacy`) — the frozen API the older chains import. Both are
  installed; scripts import one or the other by design.
- `ctx_to_lora` — the hypernetwork itself. A **separate project**, not vendored here:
  clone it and point `workflow/config.yaml:doc_to_lora_src` at its `src/`. Without it
  the CPU rules still run; gene extraction and decoding do not.
- `hyper_llm_modulator` (Text-to-LoRA) — the same arrangement for the
  hypernetwork-adapter baseline; set `t2l_src`. It loads in the main environment, since
  the released `hypermod.pt` is a plain state dict.
- `flash-attn` is only needed by the batched Qwen extractor; install it after the env
  (`pip install flash-attn==2.7.4.post1 --no-build-isolation`).
- Two baselines need a Python environment the hypernetwork cannot share, and each gets
  its own: EmbeddingGemma needs `transformers>=4.56` against the pinned 4.51.3 here, so
  `rule setup_emgemma_venv` builds `.venv-emgemma`; the `vec2text` inverter runs from
  `.venv-vec2text` (`workflow/config.yaml:vec2text_python`).

Secrets go in `.env` (gitignored): `HF_TOKEN` for the gated base models,
`OPENROUTER_API_KEY` for the judge rules.

## Data you must obtain yourself

| What | Where | Why it is not here |
|---|---|---|
| APS corpus (644k papers, metadata + abstracts) | request from [APS](https://journals.aps.org/datasets) | licensed; redistribution not permitted |
| OpenAlex snapshot tables | [openalex.org/data-dump](https://openalex.org/data-dump) | ~100 GB of upstream data |
| S2AND benchmark | [AllenAI S2AND](https://github.com/allenai/S2AND) — `rule s2and_download` fetches it | upstream distribution |
| Doc2LoRA checkpoints (Gemma-2-2B, Mistral-7B, Qwen3-4B hypernetworks) | the doc-to-lora project release | model weights, tens of GB |
| ICAE baseline weights | [ICAE](https://github.com/getao/icae) | third-party weights |
| Text-to-LoRA source and `hypermod.pt` | [SakanaAI/text-to-lora](https://github.com/SakanaAI/text-to-lora) | third-party project and weights |
| `vec2text` GTR corrector | [vec2text](https://github.com/jxmorris12/vec2text) | third-party weights, isolated venv |

Archived **intermediates** (per-unit score pools, adapters, benchmark embeddings) are on
Zenodo:

```bash
python scripts/fetch_artifacts.py results        # ~224 MB, 1227 files -> every table rebuilds on a CPU
python scripts/fetch_artifacts.py --verify       # re-check what is already on disk
snakemake paper_assets -j4 --rerun-triggers mtime   # 29 CPU jobs: tables + figures
```

The script downloads, unpacks into the repository, and verifies every file against
`data/ARTIFACTS.tsv` by SHA-256. Nothing is placed by hand — the manifest lists each
file's destination, and that is where it lands. REPRODUCE.md has the tier table, the
per-chain destinations, and the note on reading older bundles onto the current layout.
`scripts/make_artifact_bundle.py` rebuilds the bundles from a completed run and
`scripts/zenodo_upload.py` deposits them.

The 644k-paper APS gene matrices (~137 GB across three encoders) are **not** distributed:
they exceed a Zenodo record and are a deterministic function of the corpus plus the
published checkpoints. Rebuild them with `snakemake all_embeddings`.

## Layout

```
Snakefile                 24 rule files, one target per reported asset
workflow/rules/*.smk      one file per chain; its header says which result it makes
workflow/scripts/         every script the rules call, in one directory
    bench_data.py           where the data lives, and where each chain writes
    text_encoders.py        one build_* per text baseline, one encode() interface
    make_sample.py          the synthetic corpus; check_sample.py grades it
workflow/plot/            figure and table generators
libs/doc2lora, libs/legacy  the two importable APIs
scripts/                  artifact bundling, upload, and fetching
data/                     everything the workflow writes (gitignored)
data/ARTIFACTS.tsv        Zenodo manifest (tier, path, bytes, sha256)
```

Scripts are read-only and outputs are not: `workflow/scripts` holds code, `data/` holds
results, and `bench_data.out_dir` is the single answer to where a chain writes.

## What is compared against what

The paper's claim is about *decodability*, so the comparison set is not only text
encoders. Three kinds of method appear:

- **Text encoders**, which give a vector and nothing else: `all-mpnet` (SBERT),
  `SPECTER2`, `INSTRUCTOR`, `EmbeddingGemma`, `gte-large`. Built by
  `text_encoders.py`, driven by `baselines.smk` and `s2and.smk`.
- **Methods you can read back**: `ICAE` (compression into 128 memory slots), `vec2text`
  (an inverter over GTR), `T2L` (a hypernetwork that generates an adapter from a frozen
  `gte-large` vector, run at all three points in its pipeline that could be called "the
  embedding"), and `ActPatch` (training-free activation patching of Qwen3-4B hidden
  states — the one baseline needing no third-party weights: `snakemake actpatch`).
- **An LLM reading the documents directly** — in-context labelling, `KeyLLM`, and
  `BERTopic`'s c-TF-IDF keywords — which set the ceiling the vector methods are measured
  against.

The invertible citation transform $g_\theta$ is applied on top of frozen idea genes, per
field (`kron_adapter.smk`) and as a single general OpenAlex transform
(`general_adapter.smk`), so the raw and transformed rows come from the same embeddings.

## Citation

See `CITATION.cff`. The manuscript is under review; cite the preprint until it appears.
