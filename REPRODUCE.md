# Result → code map

Every figure, table, and quoted number the paper reports, with the rule that produces
it. The manuscript is not in this repository; the paths below are what the rules write
under `figs_dir` (`results/figs` by default), named as the paper reads them.

Status legend:

- ✅ **wired** — `snakemake paper_assets` builds it.
- 🔶 **typed** — a rule produces the numbers, and the paper states them as prose or as
  a hand-built table. Behind its own target, not `paper_assets`.
- ✋ **manual** — hand-drawn by design; no rule should exist.

## Assets the paper reads

| Asset | Producer | Status |
|---|---|---|
| `similarity_benchmarks.tex` (Tab. similarity, Sec. results) | `uncertainty.smk:tab_similarity_benchmarks` | ✅ |
| `encoder_matrix.tex` (Tab. per-encoder, App. tables) | same rule (third output) | ✅ |
| `pacs-clustering.pdf` (Fig. cluster-labels, Sec. results) | `fig2_pacs_clustering.smk:fig2_pacs_clustering` | ✅ |
| `mixing_decode.tex` (Tab. mixing-decode, Sec. results) | `fig2_pacs_clustering.smk:fig2_mixing_table` | ✅ |
| `hierarchy_rows.tex` (Tab. hierarchy-labels, App. tables) | `baseline_trees.smk:bt_hierarchy_rows` | ✅ |
| `prompt_sensitivity.tex` (Tab. prompt-sensitivity, App.) | `groupc_psens.smk:psens_score` | ✅ |
| `psens_edge_curves.pdf` (Fig. psens-edge, App.) | `simplex_kwgrid.smk:kg_psens_score` | ✅ |
| `incoherent_control.tex` (Tab. incoherent-control, App. cluster labels) | `groupc_incoherent.smk:incoh_score` | ✅ |
| `symmetric_raw_scores.tex` (Tab. symmetric-raw, App. symmetric-adapter) | `groupc_s2and.smk:symmetric_raw_table` | ✅ |
| `symmetric_adapter_summary.tex` (App. symmetric-adapter) | `groupc_s2and.smk:gcb_headtohead_table` | ✅ |
| `symmetric_adapter_gain.pdf` (Fig. symmetric-gain, App.) | `groupc_s2and.smk:fig_symmetric_gain` | ✅ |
| `collab_per_window.tex` (App. tables) | `groupc_s2and.smk:collab_per_window` | ✅ |
| Fig. method (Sec. intro) | hand-drawn | ✋ |

That is every `\input` and `\includegraphics` in the merged `paper/iclr2026`, and
`paper_assets` builds all twelve. Three rules write a further artifact the paper does
not read — `similarity_benchmarks.pdf`, `similarity_task_transform.tex`,
`symmetric_adapter_values.tex` — kept because they are rule outputs a rebuild can be
diffed against.

### Baselines, and where each one is reproduced

| Baseline | Chain | Reported in |
|---|---|---|
| SBERT / SPECTER2 / Instructor / EmbeddingGemma / GTE / BGE | `baselines.smk`, `s2and.smk`, `groupc_bench.smk:gcb_embed` | Tab. similarity, App. symmetric-adapter |
| `ICAE` | `icae.smk` (slots + its own transform) | App. symmetric-adapter, cluster labels, fusion edge, length-dial |
| `T2L` | `t2l.smk` | cluster labels, fusion edge |
| `vec2text` | `baseline_trees.smk:bt_vec2text`, `abstraction_walk.smk:aw_vec2text` | cluster labels, Tab. radial-vec2text |
| `BERTopic` | `baseline_trees.smk:bt_bertopic*`, `bt_bertopic_judge` | App. BERTopic, Tab. label-eval |
| `KeyLLM` / in-context | `baseline_trees.smk` | cluster labels |
| `ActPatch` | `actpatch.smk` | App. actpatch |

ICAE's retrieval-evaluation half is deliberately absent: no reported row is an ICAE
retrieval score. What the paper reads from ICAE is its embeddings, through the
symmetric-adapter control.

### The citation sample the reported transform was trained on

`triplets_1x.parquet` is the one artifact the workflow cannot derive. A later
data-scaling run overwrote the unsuffixed `triplets.parquet` with a different draw --
not a superset; 29 of the reported 42,332 tuples survive in it -- so the reported
sample is carried by name, ships in the `results` tier (1.3 MB, ids only), and is
pointed at by `workflow/config.yaml:ga_triplets`.

Its 167,111-paper text pool is NOT shipped and does not need to be: `sample_edges.py`
writes exactly the papers the triplets name, so `ga_pool_text` rebuilds
`pool_text_1x.parquet` from the ids and the OpenAlex tables. Everything downstream --
`g_theta`, the ICAE transform, every arm of the symmetric control -- then trains on the
sample the paper reports.

The adapters are named for that sample (`adapter_general_qwen_1x.pt`) because the bare
name is a trap: on the machine that produced the paper `adapter_general_qwen.pt` is
byte-identical to the 2x adapter, and applying it reproduces the published vectors at
cosine .005 instead of 1.0.

## Testing the workflow without the corpora

`snakemake sample_check_judged -j4` builds a small synthetic field and runs the real
scoring scripts over it, judge panel included, in about twenty seconds on a CPU. The
method ordering is planted by `workflow/scripts/make_sample.py` and recovered by
`workflow/scripts/check_sample.py`, so a pipeline that drops a method or unpairs the
score columns fails the check. See `workflow/rules/sample.smk`, and the README for what
it does and does not cover.

### Figure: cluster labels and mixtures, panel by panel

The figure is six panels drawn from six small JSON/CSV summaries, each the cheap
end of an expensive run. Drawing the figure from them is seconds of CPU; producing
them is the GPU work named in the last column. They live under `data/`, so a clone
either rebuilds them through those rules or unpacks them from the artifact bundle.

| Panel | Shows | Cache | Produced by |
|---|---|---|---|
| (a) | fuzzy label overlap per PACS node | `label_eval_metric1.json` | `baseline_trees.smk:bt_label_eval_metric1` |
| (b) | judge round robin between methods | `label_eval_metric4.json` | `baseline_trees.smk:bt_label_eval_metric4` (`snakemake label_eval`) |
| (c) | PACS breadth against tree depth | `length_vs_breadth.csv`, `qwen_fullrank_field23.json` | `baseline_trees.smk:bt_length_dial`, `bt_decode_fullrank` |
| (d) | Wikipedia abstraction walk | `abstraction_walk.json` | `abstraction_walk.smk:aw_doc2lora` |
| (e) | actual against ideal mixing weight | `pair_axis_metrics_pairaxis.json` | `simplex_kwgrid.smk:kg_metrics` |
| (f) | verbatim copy rate along the edge | `pair_axis_copyrate_pairaxis_L1.json`, `pair_axis_t2l_e1_pairaxis.json` | `simplex_kwgrid.smk:kg_copyrate`, `fig2_pacs_clustering.smk:fig2_t2l_edge_metrics` |

The panel (b) judge cache is a **param**, not an input, so asking for the figure
never spends OpenRouter budget; after a fresh judge run, re-plot with
`snakemake figure2 --forcerun fig2_pacs_clustering`.

Snakemake links rules by path, so asking for the figure through the workflow
schedules the producer of every summary it declares — on a clean tree that is the
whole GPU chain. Once the summaries exist, redraw without the workflow by calling
the two scripts directly:

```bash
python workflow/plot/fig_pacs_clustering.py --out results/figs/pacs-clustering.pdf
python workflow/scripts/fig2_mixing_table.py --out results/figs/mixing_decode.tex
```

Both are CPU-only and take seconds.

## Unpacking the archived intermediates

Path 1 of the README — every reported table rebuilt on a CPU — reads score pools,
adapters, and benchmark vectors that a GPU produced once. They are published as a
Zenodo record and fetched by one command:

```bash
python scripts/fetch_artifacts.py results          # ~192 MB, 219 files
python scripts/fetch_artifacts.py results s2and    # + ~12.3 GB, 50 files
python scripts/fetch_artifacts.py --verify         # re-check what is already on disk
python scripts/fetch_artifacts.py --record <ID> results   # pin a record
```

The script resolves the record, downloads the tarball for each tier you name, unpacks
it into the repository, and checks every file against `data/ARTIFACTS.tsv` by SHA-256.
There is nothing to place by hand: `data/ARTIFACTS.tsv` lists the destination of every
archived file, and that is where it lands.

| Tier | Size | What it holds | What it unlocks |
|---|---|---|---|
| `results` | ~192 MB | per-unit score pools, the bootstrap replicates, the trained adapters, the PACS node set, the label decodes and judge verdicts, the published tables | `snakemake paper_assets` on a CPU |
| `s2and` | ~12.3 GB | the disambiguation corpora's vectors, per encoder | `snakemake s2and uncertainty` — re-score from vectors |
| `aps` | ~137 GB | the 644k-paper gene matrices | **not distributed**: `snakemake all_embeddings` |

Where things land, by chain:

```
data/uncertainty/pools/      per-unit score pools; bootstrap reads these
data/uncertainty/            uncertainty_summary.csv -> both similarity tables
data/kron/  data/general_adapter/     trained transforms and per-task score tables
data/s2and/proc/<dataset>/   disambiguation vectors (the s2and tier)
data/labels/                 node labels, decodes, judge verdicts, fullrank means
data/pacs/results/           the PACS node set every labelling table is built on
data/groupc/                 the appendix controls
results/figs/                the published tables, so a rebuild can be diffed
```

The `aps` tier is left out on purpose: it exceeds a Zenodo record and is a
deterministic function of the corpus plus the published checkpoints.

**Layout note.** The bundles were packed when each chain kept its outputs next to its
own code, under a dated `exps/` directory. This workflow separates the two, so
`fetch_artifacts.py` rewrites each member onto the current layout as it unpacks
(`LEGACY_LAYOUT` in that file). An archived
`exps/2026-06-10-uncertainty/pools/np_aps_qwen.parquet` therefore arrives as
`data/uncertainty/pools/np_aps_qwen.parquet`, which is what the manifest lists and
what the rules read. `scripts/make_artifact_bundle.py` writes the current layout
directly, so a bundle rebuilt from a completed run needs no rewriting.

Having unpacked a tier, confirm the workflow agrees that it is satisfied:

```bash
snakemake -n paper_assets --rerun-triggers mtime    # should collapse to the table rules
```

## Numbers typed into the text

The manuscript states these as prose or as a hand-built `tabular`. The rule
produces the numbers; the transcription is manual.

| Number / table | Producer | Target | Status |
|---|---|---|---|
| Tab. label-eval (App. cluster labels): fuzzy overlap, word counts, pairwise wins | `baseline_trees.smk:bt_label_eval_*` → `label_eval_summary.md`, `label_eval_metric4.json` | `snakemake label_eval` | 🔶 |
| App. recipe-fusion: the two source recipes, the decoded midpoint, and the cheese answer, all quoted verbatim | `recipe_fusion.smk` → `data/recipe_fusion/recipe_fusion.json`, `<figs_dir>/recipe_fusion.tex` | `snakemake recipe_fusion` | 🔶 |
| App. actpatch: identification rate and MRR, midpoint gains, the $28$-node comparison | `actpatch.smk` → `score_compare.json`, `midpoint_score.json`, `arith_m3/m4.json` | `snakemake actpatch` (add `actpatch_judged` for the LLM panel) | 🔶 |
| T2L: the three embedding positions, the label arm, the fusion arm and its two validity gates | `t2l.smk` → `t2l_{gte,hidden,dw}_labels.json`, `t2l_fusion_score.json` | `snakemake t2l` | 🔶 |
| Sec. fusion / App. prompt-sensitivity: the slopes ($1.11$–$1.20$, $1.19$–$1.24$, $0.29$–$0.41$) and copy rates | `simplex_kwgrid.smk:kg_psens_score` → `psens_edge.json` | in `paper_assets` | 🔶 |
| Temporal hardening: "next-paper AUC varies by at most .005", the 2016 topic split (App. datasets, #72) | `groupc_bench.smk:gcb_bootstrap` → `<figs_dir>/groupc_temporal.tex` | `snakemake temporal_hardening` | 🔶 |
| Train/eval overlap: "<1% of evaluation papers", "at most 1.24% of edges" (App. datasets, #22) | `general_adapter.smk:ga_leakage_overlap` | `snakemake leakage` | 🔶 |
| Pooling ρ (mean-over-rank vs full tensor, Sec. methods) | `pooling_validation` → `data/aps/pooling_spearman.csv` | in `paper_assets` | 🔶 |
| Tab. "Decoding clusters and mixtures" (Sec. results) — verbatim decodes | `baseline_trees.smk:bt_decode_fullrank` → `qwen_fullrank_field23.json` | `snakemake paper_assets` | 🔶 |

`label_eval`, `actpatch_judged`, and the incoherent-cluster control call an LLM judge
panel through OpenRouter, so they need `OPENROUTER_API_KEY` and cost money;
`paper_assets` deliberately stops short of the judge panel (metric 4). Judge
verdicts are not bit-reproducible — the panel is a measurement instrument with its
own variance, not a deterministic function.

Every rule reaches a reported result. The reachability is checked, not assumed: each
script in `workflow/scripts` is named by a rule or imported by one that is, and each
rule is scheduled by one of the targets above.

### A note on `--rerun-triggers`

Snakemake's default trigger set includes `params`, `input` and `code`, so a rule
whose output this repository ships can still be scheduled on a fresh clone that has
no `.snakemake` provenance for it. `snakemake -n paper_assets --rerun-triggers mtime`
asks the narrower question — what is actually out of date on disk — and is the
setting these job counts assume.

## The baselines, and where each one lives

Every method the manuscript compares against is built here. They fall into three
groups by what they need.

**Text encoders** are embedded by one script, `workflow/scripts/text_encoders.py`,
which holds one `build_*` function per encoder behind a shared `encode(texts)`
interface, and driven by `baselines.smk` for the field corpora and by
`s2and.smk` for the disambiguation corpora:

| Baseline | Model | Note |
|---|---|---|
| `sbert` | `sentence-transformers/all-mpnet-base-v2` | also the scorer behind the decode metrics |
| `specter2` | `allenai/specter2` | citation-trained; `SPECTER` on the disambiguation rows |
| `instructor` | `hkunlp/instructor-large` | instruction-prefixed |
| `embeddinggemma` | `google/embeddinggemma-300m` | needs `transformers>=4.56`, so it embeds in an isolated `.venv-emgemma` that `rule setup_emgemma_venv` builds; its batch size is the `emgemma_batch_size` key |
| `gte` | `Alibaba-NLP/gte-large-en-v1.5` | loaded with `trust_remote_code` |
| `vec2text` | GTR-base corrector | an inverter, not an encoder: it takes no prompt, so it is read back as a reconstruction. Isolated `.venv-vec2text` |

**Compression baselines** decode from a bottleneck the way \doctolora decodes from
an adapter, which is what makes them the interesting comparison:

| Baseline | Where | Note |
|---|---|---|
| `ICAE` | `baseline_trees.smk:bt_icae_raw` (labels), `simplex_kwgrid.smk` (fusion edge), `abstraction_walk.smk:aw_icae` (magnitude sweep) | Mistral-7B, 128 memory slots of dim 4096. Third-party weights and source tree: set `icae_weights`, `icae_code_dir`, `icae_base_model`. It is a **decoding** comparison only — no reported retrieval row is ICAE, so no retrieval chain for it exists here |
| `T2L` | `t2l.smk` | Text-to-LoRA: a hypernetwork that expands a frozen `gte-large` vector into a LoRA. Separate clone, like `doc-to-lora`: set `t2l_src` |
| `ActPatch` | `actpatch.smk` | training-free activation patching of Qwen3-4B hidden states; needs no third-party weights |
| `KeyLLM` / in-context | `baseline_trees.smk` | an LLM reading the documents, for the cluster-label comparison |

**T2L runs three arms, not one.** Three points in its pipeline can each be called
"the embedding", and reporting only the first would be the shortcut a reviewer
would find: `e0` the frozen `gte-large` output (1,024-dim), `e1` the TaskEncoder
output (64-dim, the only document-dependent learned layer), and `e2` the generated
LoRA factors (3,407,872-dim, the structural analogue of an idea gene). `t2l.smk`
gates every reportable arm behind two checks that must pass first — `t2l_sanity`
(do different documents give different adapters at all) and `t2l_validity` (does
the pipeline reproduce T2L's intended behaviour on a task it was trained on).
Every method receives the paper's title and abstract verbatim, T2L included, so
T2L runs outside its training distribution; that is a stated interpretation limit
of the comparison rather than something the workflow patches around.

## Fixes this repository carries over the development tree

Trimming the workflow surfaced three defects that a fresh checkout hits and an
in-place working tree does not. All three are fixed here:

1. **`data_dir: "./data/"` split the DAG.** Snakemake treats `./data/x` and
   `data/x` as different files, and the genkron rules emit the un-prefixed form
   while everything else used `DATA_DIR`. The adapter chain therefore had no
   producer for its own inputs on a clean tree. `data_dir` is now `data/`.
2. **A global wildcard constraint blocked a baseline.** `s2and.smk` declares
   `method="sbert|instructor|embeddinggemma|gte"` at module level, which silently
   prevented `embed_text_baseline` from ever producing `baseline_specter2.npz`.
   `baselines.smk` now sets its own per-rule constraints.
3. **Three scripts that produce manuscript assets were gitignored**
   (`compute_qwen_fullrank_means.py`, `decode_fullrank_field23.py`,
   `make_pacs_label_tables.py`), so a clone could not build the cluster-label
   tables. They are tracked here.

Three more inputs had no producer at all — they existed on the development
machine as the residue of manual runs, so the DAG only closed there:

4. **The APS text baselines.** `baselines.smk` is wildcard-scoped to
   `data/fields/...`; nothing produced `data/aps/embeddings/baseline_specter2.npz`
   or `baseline_instructor.npz`, which `unc_pool_field` and the Kron chain both
   read. `rule embed_aps_text_baseline` now does, via the same embedder (patched
   to accept the APS `aps_paper_id` spelling).
5. **`data/aps/paper_text_pid.parquet`.** The label chains expect the APS table
   keyed on `paper_id`. Verified to be a pure column rename of
   `paper_text.parquet` (identical row count, identical ids) and now produced by
   `rule aps_text_pid`.
6. **The benchmark-subset vectors.** `bench.smk` models its slicer with a marker
   output, because which files it writes depends on which embeddings exist, while
   `groupc_bench.smk` declared the sliced `.npz` directly as inputs — so the
   temporal-hardening chain was unreachable. `gcb_apply` now depends on the
   marker and takes the vector path as a param.

Two further changes make previously unreachable steps reproducible:

7. **The PACS node set is now a rule.** `paper_groups.parquet` / `groups.parquet`
   — the node set every labelling table is built on — could only be produced by a
   Docker-specific sub-Snakefile, so they looked like frozen artifacts.
   `workflow/rules/pacs_groups.smk` wires them, rendering the sub-config from
   `workflow/config.yaml` so paths have one source of truth.
8. **Hardcoded absolute paths removed.** Several scripts carried an author's home
   directory on `sys.path`; they now resolve the repository root from `__file__`.

Two more classes of defect came out of reading the scripts rather than the DAG:

9. **Hardcoded data roots.** Eight scripts read `/data/datasets/aps/preprocessed/...`
   and `/data/projects/gravity-of-ideas/...` as string literals, with no way to
   override them. The `.smk` files all wrap such paths in `config.get(key, default)`,
   so this was invisible from the rule layer and invisible to a dry run — it only
   fails at execution time, on any machine whose data sits elsewhere. All eight now
   resolve through `workflow/scripts/bench_data.py`, which reads
   `$ENV_VAR` → `workflow/config.yaml` → built-in default, using the same keys the
   Snakefile does. `python workflow/scripts/bench_data.py` prints what resolved and
   from where.

   `bench_data.py` also absorbs the two loaders (`load`, `collab_net`) that fourteen
   scripts used to import from `eval_collab.py`. That file mixed a shared
   data-locating layer with a first-generation collaboration evaluation that no rule
   called and that the manuscript does not report — the reported collaboration
   numbers come from `eval_collab2.py`. The dead evaluation is deleted, so there is
   now one evaluator and one data-locating module instead of a confusing v1/v2 pair.

10. **`groupc/fid_score.py` could not be parsed at all** — an f-string closed a
    literal LaTeX brace with a single `}`. The file ships from a chain that was
    deliberately not run (#95 is answered by citing the doc-to-lora paper), which is
    how a syntax error survived. Fixed and `compileall` now passes over the whole
    tree.
11. **The pair-axis ICAE decoder has its own module.** `decode_absfollow_icae.py`
    reached across into an exploratory experiment directory for the ICAE loader it
    needs, and that directory is not part of the manuscript's closure. The loader is
    `workflow/scripts/icae_slots.py` here: a thin wrapper over
    `icae_lib.load_icae` that resolves the third-party weights, source tree, and base
    model through `workflow/config.yaml`, so the ICAE arm of the edge decode has one
    dependency instead of a chain of sideways imports.
12. **Code and results are separate directories.** A dated experiment folder used to
    hold both, so a script wrote its outputs next to itself and a rule could point at
    a path that an archived experiment had taken with it. Scripts now live in
    `workflow/scripts` and every chain writes under `data/`, resolved by
    `bench_data.out_dir` and redirectable per chain with `$<TOPIC>_OUT` -- which is
    what lets the sample corpus run the real scripts without touching a real result.

Scripts that no rule reaches (`eval_next_paper.py`, `eval_yearmean.py`,
`year_balanced_collab.py`, `specter2_embed.py`) are kept for provenance and now say
so in their own docstrings, so they cannot be mistaken for pipeline steps.

## What was left out, and why

This repository carries 24 rule files. The chains that are absent produced results
that never reached the manuscript: simplex fusion and the fusability sweeps,
topological (TDA) hole-finding, the idea-gene GA, cross-domain analogy, arXiv field
splits, interpolation demos, and the multi-agent holes study.

## Cold-start job counts

From nothing but the corpora and checkpoints (`snakemake -n <target>`):

| Target | Jobs | Heaviest step |
|---|---|---|
| `paper_assets` | 329 | 3 × 644k-paper gene extraction (GPU-days) |
| `groupc_s2and` | 190 | the symmetric control over all 14 benchmarks |
| `uncertainty` | 133 | 1000× bootstrap over every paired score pool |
| `general` | 115 | the general adapter + applying it everywhere |
| `temporal_hardening` | 78 | retraining the transform on pre-2018 citations |
| `s2and` | 62 | gene extraction for five disambiguation corpora |
| `kron` | 43 | per-field citation adapters |
| `figure2` | 41 | the pair-axis edge decodes behind panels (e), (f) |
| `label_eval` | 21 | LLM judge panel (OpenRouter, costs money) |
| `icae_embeddings` | 15 | ICAE slots for every benchmark subset |
| `actpatch_judged` | 13 | 500 decodes per arm, then the judge panel |
| `pair_axis` | 13 | 250 pairs × 13 points × 2 decoders |
| `fields` | 11 | economics + psychology corpora and genes |
| `bench_subsets` | 11 | slicing every embedding to the touched ids |
| `t2l` | 9 | the hypernetwork baseline's label and fusion arms |
| `sample_check_judged` | 7 | nothing: 20 seconds on a CPU |
| `pacs_groups` | 5 | the PACS node set |
| `abstraction_walk_all` | 4 | the ICAE / vec2text magnitude sweeps |
| `recipe_fusion` | 3 | one Mistral-7B decode of two recipes |

Every one of these was checked with `snakemake -n` against an **empty** `data/`
directory, so the counts are true cold-start figures: the only inputs assumed to
exist are the licensed corpora and the checkpoints named in
`workflow/config.yaml`.

With the `results` artifact bundle unpacked, the table-building tail of
`paper_assets` runs on a CPU in minutes.

Note that Snakemake's default trigger set includes `params`, `input` and `code`, so a
clone with no `.snakemake` provenance can schedule more than is strictly out of date.
`snakemake -n <target> --rerun-triggers mtime` asks the narrower question.
