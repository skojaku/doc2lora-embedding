# Doc2LoRA idea genes — reproduction repository

Code and data manifest for reproducing every result in *"Doc2LoRA Provides
Decodable Representations of Scientific Ideas"* (`paper/iclr2026/`).

A **Doc2LoRA idea gene** is what you get when a hypernetwork reads a paper and
emits a small LoRA adapter for it: the adapter's latent tensor becomes the
paper's embedding. Unlike an ordinary sentence embedding, that vector can be
loaded back into the language model, so any point in the space — including a
point you constructed by averaging or interpolating — can be *decoded* back into
text. This repository contains the pipeline that measures how well those vectors
work as embeddings and what their decodes look like.

The workflow is trimmed to the **dependency closure of the manuscript**: the main
text, its appendices, and nothing else. Exploratory chains that never reached the
PDF are not here.

---

## What you can reproduce, and what it costs

There are four entry points, in increasing order of cost. Pick the cheapest one
that answers your question.

| # | You want to | Command | Needs | Time |
|---|---|---|---|---|
| 0 | Redraw the **main figure** from the caches in this repository | `python workflow/plot/fig_pacs_clustering.py --out figs/pacs-clustering.pdf` | CPU, no download | seconds |
| 1 | Rebuild every **table** in the paper from archived scores | `python scripts/fetch_artifacts.py results` then `snakemake paper_assets -j4` | CPU, ~130 MB download | minutes |
| 2 | Re-score the **name-disambiguation** rows from the vectors | `python scripts/fetch_artifacts.py results s2and` then `snakemake s2and uncertainty -j4` | CPU, +12 GB download | hours |
| 3 | Re-derive **everything from the corpora**, genes included | `snakemake paper_assets -j4` | 1–4 GPUs (≥48 GB total), ~200 GB disk, licensed APS corpus | GPU-days |

Path 1 is the one to start with: it checks that the published numbers follow from
the archived per-unit scores, without a GPU and without the licensed corpus.

Path 0 is cheaper still and needs nothing at all. Figure 2, of cluster labels and
mixtures, is drawn from six small caches — the fuzzy label scores, the judge round
robin, the breadth/depth table, the abstraction walk, and the two pair-axis
summaries — and all six are committed here. `REPRODUCE.md` maps each panel to its
cache and to the rule that regenerates that cache from scratch.

```bash
# the whole of path 1
./install.sh
cp workflow/config.template.yaml workflow/config.yaml   # edit the paths
python scripts/fetch_artifacts.py results --record <ZENODO_ID>
snakemake paper_assets -j4
snakemake paper          # -> paper/iclr2026/main.pdf
```

---

## Step 0 — configuration (mandatory)

`Snakefile` reads `workflow/config.yaml`, which is gitignored so your machine's
paths never land in git. Copy the template and edit it:

```bash
cp workflow/config.template.yaml workflow/config.yaml
$EDITOR workflow/config.yaml     # every "<-- set me" line
```

Ten keys are read with a hard `config[...]` and must exist or Snakemake fails
while parsing: `data_dir`, `paper_dir`, `aps_paper_table`,
`openalex_paper_table`, `openalex_abstracts`, `mistral_checkpoint_path`,
`qwen_checkpoint_path`, `icae_weights`, `shard_size`, `gpu_ids`. Everything else
has a default, and the template's values are the ones the manuscript used.

Then check the graph without running anything:

```bash
snakemake -n paper_assets      # prints the DAG; 195 jobs from a cold start
```

## Environment

```bash
./install.sh          # conda env "doc2lora" + editable installs of libs/
```

- `libs/doc2lora` — the current embed / arithmetic / decode API.
- `libs/legacy` (`doc2lora_legacy`) — the frozen API the `exps/` chains import.
  Both are installed; scripts import one or the other by design.
- `ctx_to_lora` — the hypernetwork implementation itself. It is a **separate
  project**, not vendored here. Clone it and point
  `workflow/config.yaml:doc_to_lora_src` at its `src/`. Without it the CPU rules
  still run; the gene-extraction and decode rules do not.
- `hyper_llm_modulator` (Text-to-LoRA) — a separate project on the same footing,
  used by the hypernetwork-adapter baseline. Clone it and point
  `workflow/config.yaml:t2l_src` at its `src/`. It loads in the main environment:
  the released `hypermod.pt` is a plain state dict, so T2L's own `transformers` /
  `vllm` pins are only needed by its training harness.
- `flash-attn` is only needed by the batched Qwen extractor; install it after the
  env (`pip install flash-attn==2.7.4.post1 --no-build-isolation`).
- Two baselines need a Python environment the hypernetwork cannot share, and each
  gets its own: EmbeddingGemma needs `transformers>=4.56` against the pinned
  4.51.3 here, so `rule setup_emgemma_venv` builds `.venv-emgemma`; the `vec2text`
  inverter runs from `.venv-vec2text` (`workflow/config.yaml:vec2text_python`).

Secrets go in `.env` (gitignored): `HF_TOKEN` for the gated base models,
`OPENROUTER_API_KEY` for the LLM-judge rules (`label_eval`, and the
incoherent-cluster control if `incoh_use_judges: true`).

## Data you must obtain yourself

| What | Where | Why it is not here |
|---|---|---|
| APS corpus (644k papers, metadata + abstracts) | request from [APS](https://journals.aps.org/datasets) | licensed; redistribution not permitted |
| OpenAlex snapshot tables | [openalex.org/data-dump](https://openalex.org/data-dump) | ~100 GB of upstream data |
| S2AND benchmark | [AllenAI S2AND](https://github.com/allenai/S2AND) — `rule s2and_download` fetches it | upstream distribution |
| Doc2LoRA checkpoints (Gemma-2-2B, Mistral-7B, Qwen3-4B hypernetworks) | see the doc-to-lora project release | model weights, tens of GB |
| ICAE baseline weights | [ICAE](https://github.com/getao/icae) | third-party weights |
| Text-to-LoRA source and `hypermod.pt` | [SakanaAI/text-to-lora](https://github.com/SakanaAI/text-to-lora), weights on Hugging Face | third-party project and weights |
| `vec2text` GTR corrector | [vec2text](https://github.com/jxmorris12/vec2text) | third-party weights, isolated venv |

Archived **intermediates** (per-unit score pools, adapters, benchmark
embeddings) are on Zenodo and fetched by `scripts/fetch_artifacts.py`; see
`data/ARTIFACTS.tsv` for the exact file list with sizes and SHA-256 digests.
`scripts/make_artifact_bundle.py` rebuilds those bundles from a completed run and
`scripts/zenodo_upload.py` deposits them (as a draft; publishing stays manual).

The 644k-paper APS gene matrices (~137 GB across three encoders) are **not**
distributed: they exceed a Zenodo record and are a deterministic function of the
corpus plus the published checkpoints. Rebuild them with
`snakemake all_embeddings`.

## Layout

```
Snakefile                 the trimmed workflow: 23 rule files, one target per asset
workflow/rules/*.smk      one file per chain (genes, adapters, baselines, figures)
workflow/scripts/         the steps the rules call
workflow/scripts/bench_data.py  where the data lives (env -> config -> default)
workflow/scripts/text_encoders.py  one build_* per text baseline, one encode() interface
workflow/plot/            figure and table generators (PDF and LaTeX fragments)
libs/doc2lora, libs/legacy  the two importable APIs
exps/<date>-<topic>/      the per-experiment scripts the rules reach into
paper/iclr2026/           the manuscript (main.tex, sections/, main.bib)
figs/                     the assets the manuscript reads
scripts/                  artifact bundling, upload, and fetching
data/ARTIFACTS.tsv        Zenodo manifest (tier, path, bytes, sha256)
```

`REPRODUCE.md` is the result → code map: for every figure, table, and quoted
number, which rule produces it and whether that rule is wired.

## What is compared against what

The manuscript's claim is about *decodability*, so the comparison set is not only
text encoders. Three kinds of method appear:

- **Text encoders**, which give a vector and nothing else: `all-mpnet` (SBERT),
  `SPECTER2`, `INSTRUCTOR`, `EmbeddingGemma`, and `gte-large`. Built by
  `workflow/scripts/text_encoders.py` and driven by `baselines.smk` (field corpora)
  and `s2and.smk` (author-name disambiguation).
- **Methods you can read back**, which is where the interesting comparison is:
  `ICAE` (compression into 128 memory slots), `vec2text` (an inverter over GTR),
  `T2L` / Text-to-LoRA (a hypernetwork that generates an adapter from a frozen
  `gte-large` vector, run at all three points in its pipeline that could be called
  "the embedding"), and `ActPatch` (training-free activation patching of Qwen3-4B
  hidden states).
- **An LLM reading the documents directly** — in-context labelling and `KeyLLM` —
  which sets the ceiling the vector methods are measured against.

`ActPatch` is the one baseline that needs no third-party weights at all: it reads
hidden states out of the same Qwen3-4B the idea genes decode through, so the two
arms differ only in where the injected vector comes from. `snakemake actpatch`
runs it end to end.

The invertible citation transform $g_\theta$ is applied on top of frozen idea
genes, per field (`kron_adapter.smk`) and as a single general OpenAlex transform
(`general_adapter.smk`), so both the raw and transformed rows in the tables come
from the same embeddings.

## Citation

See `CITATION.cff`. The manuscript is under review; cite the preprint until it
appears.
