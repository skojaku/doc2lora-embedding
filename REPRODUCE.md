# Result → code map

Every figure, table, and quoted number in `paper/iclr2026/main.tex` (main text
**and** appendices), with the rule that produces it.

Status legend:

- ✅ **wired** — a rule produces it, and `snakemake paper_assets` builds it.
- 🔶 **typed** — a rule produces the numbers, but the `.tex` in the manuscript is
  hand-assembled from them. Behind its own target, not `paper_assets`.
- ✋ **manual** — hand-drawn or hand-written by design; no rule should exist.
- ❌ **gap** — the manuscript needs it and no rule in this repository makes it.

## Assets the manuscript reads

These are the complete set of `\includegraphics` / `\input` targets in
`main.tex` and `sections/*.tex`.

| Asset | Producer | Status |
|---|---|---|
| `figs/similarity_benchmarks.tex` (Tab. similarity, Sec. results) | `uncertainty.smk:tab_similarity_benchmarks` → `workflow/plot/tab_similarity_benchmarks.py` | ✅ |
| `figs/encoder_matrix.tex` (Tab. per-encoder, App. tables) | same rule (third output) | ✅ |
| `figs/pacs-clustering.pdf` (Fig. cluster-labels, Sec. results) | `fig2_pacs_clustering.smk:fig2_pacs_clustering` → `workflow/plot/fig_pacs_clustering.py` | ✅ |
| `figs/mixing_decode.tex` (Tab. mixing-decode, Sec. results) | `fig2_pacs_clustering.smk:fig2_mixing_table` → `workflow/scripts/fig2_mixing_table.py` | ✅ |
| `paper/iclr2026/hierarchy_rows.tex` (Tab. hierarchy-labels, App. tables) | `baseline_trees.smk:bt_hierarchy_rows` → `exps/2026-06-11-baseline-trees/make_hierarchy_rows.py` | ✅ |
| `figs/prompt_sensitivity.tex` (Tab. prompt-sensitivity, App.) | `groupc_psens.smk:psens_score` | ✅ |
| `figs/psens_edge_curves.pdf` (Fig. psens-edge, App.) | `simplex_kwgrid.smk:kg_psens_score` → `exps/2026-07-02-simplex-kwgrid/psens_edge_score.py` | ✅ |
| `figs/incoherent_control.tex` (Tab. incoherent-control, App. cluster labels) | `groupc_incoherent.smk:incoh_score` | ✅ |
| `figs/doc2lora-figs.pdf` (Fig. method, Sec. intro) | hand-drawn, source `figs/doc2lora-figs.graffle` | ✋ |
| `paper/iclr2026/math_commands.tex` | local LaTeX macros | ✋ |

Three rules write a fourth artifact the manuscript does not `\input`:
`figs/similarity_benchmarks.pdf` (the same benchmark scores as a dot plot),
`figs/similarity_task_transform.tex` (Tab. similarity with the task-specific
transform column), and `figs/prompt_sensitivity_edge.tex` (the edge-prompt numbers
the appendix states in prose). They are kept because they are rule outputs, and
because a rebuild can be diffed against them.

`paper_assets` depends on exactly the ✅ rows plus `data/aps/pooling_spearman.csv`
(the Spearman ρ quoted in Sec. methods). Adding anything else to that target is a
claim the manuscript reads it. Every asset the manuscript reads has a rule;
nothing in the PDF is unreachable from this workflow.

### Figure: cluster labels and mixtures, panel by panel

The figure is six panels drawn from six caches, and each cache is the cheap
summary of an expensive run. All six caches are tracked in git, so the figure
itself is a CPU job of a few seconds.

| Panel | Shows | Cache | Produced by |
|---|---|---|---|
| (a) | fuzzy label overlap per PACS node | `label_eval_metric1.json` | `baseline_trees.smk:bt_label_eval_metric1` |
| (b) | judge round robin between methods | `label_eval_metric4.json` | `baseline_trees.smk:bt_label_eval_metric4` (`snakemake label_eval`) |
| (c) | PACS breadth against tree depth | `length_vs_breadth.csv`, `qwen_fullrank_field23.json` | `baseline_trees.smk:bt_length_dial`, `bt_decode_fullrank` |
| (d) | Wikipedia abstraction walk | `abstraction_walk.json` | `abstraction_walk.smk:aw_doc2lora` |
| (e) | actual against ideal mixing weight | `pair_axis_metrics_pairaxis.json` | `simplex_kwgrid.smk:kg_metrics` |
| (f) | verbatim copy rate along the edge | `pair_axis_copyrate_pairaxis_L1.json`, `pair_axis_t2l_e1_pairaxis.json` | `simplex_kwgrid.smk:skg_copyrate`, `fig2_pacs_clustering.smk:fig2_t2l_edge_metrics` |

The panel (b) judge cache is a **param**, not an input, so asking for the figure
never spends OpenRouter budget; after a fresh judge run, re-plot with
`snakemake figure2 --forcerun fig2_pacs_clustering`.

Snakemake links rules by path, so asking for the figure through the workflow
schedules the producer of every cache it declares — on a clean clone that is the
GPU chain behind them. To redraw the figure from the tracked caches alone, call
the two scripts directly:

```bash
python workflow/plot/fig_pacs_clustering.py --out figs/pacs-clustering.pdf
python workflow/scripts/fig2_mixing_table.py --out figs/mixing_decode.tex
```

Both are CPU-only and take seconds.

## Numbers typed into the text

The manuscript states these as prose or as a hand-built `tabular`. The rule
produces the numbers; the transcription is manual.

| Number / table | Producer | Target | Status |
|---|---|---|---|
| Tab. label-eval (App. cluster labels): fuzzy overlap, word counts, pairwise wins | `baseline_trees.smk:bt_label_eval_*` → `label_eval_summary.md`, `label_eval_metric4.json` | `snakemake label_eval` | 🔶 |
| App. recipe-fusion: the two source recipes, the decoded midpoint, and the cheese answer, all quoted verbatim | `recipe_fusion.smk` → `exps/2026-06-11-recipe-fusion/results/recipe_fusion.json`, `figs/recipe_fusion.tex` | `snakemake recipe_fusion` | 🔶 |
| App. actpatch: identification rate and MRR, midpoint gains, the $28$-node comparison | `actpatch.smk` → `score_compare.json`, `midpoint_score.json`, `arith_m3/m4.json` | `snakemake actpatch` (add `actpatch_judged` for the LLM panel) | 🔶 |
| T2L: the three embedding positions, the label arm, the fusion arm and its two validity gates | `t2l.smk` → `t2l_{gte,hidden,dw}_labels.json`, `t2l_fusion_score.json` | `snakemake t2l` | 🔶 |
| Sec. fusion / App. prompt-sensitivity: the slopes ($1.11$–$1.20$, $1.19$–$1.24$, $0.29$–$0.41$) and copy rates | `simplex_kwgrid.smk:kg_psens_score` → `psens_edge.json` | in `paper_assets` | 🔶 |
| Temporal hardening: "next-paper AUC varies by at most .005", the 2016 topic split (App. datasets, #72) | `groupc_bench.smk:gcb_bootstrap` → `figs/groupc_temporal.tex` | `snakemake temporal_hardening` | 🔶 |
| Train/eval overlap: "<1% of evaluation papers", "at most 1.24% of edges" (App. datasets, #22) | `general_adapter.smk:ga_leakage_overlap` | `snakemake leakage` | 🔶 |
| Pooling ρ (mean-over-rank vs full tensor, Sec. methods) | `pooling_validation` → `data/aps/pooling_spearman.csv` | in `paper_assets` | 🔶 |
| Tab. "Decoding clusters and mixtures" (Sec. results) — verbatim decodes | `baseline_trees.smk:bt_decode_fullrank` → `qwen_fullrank_field23.json` | `snakemake paper_assets` | 🔶 |

`label_eval`, `actpatch_judged`, and the incoherent-cluster control call an LLM judge
panel through OpenRouter, so they need `OPENROUTER_API_KEY` and cost money;
`paper_assets` deliberately stops short of the judge panel (metric 4). Judge
verdicts are not bit-reproducible — the panel is a measurement instrument with its
own variance, not a deterministic function.

Two rules survive whose output the current manuscript does not state:
`groupc_efficiency` (`snakemake efficiency`) measures throughput and index size,
which an earlier draft tabulated, and `groupc_bench` (`snakemake
temporal_hardening`) backs the decontamination claims of App. datasets. They are
kept because the claims they answer can be asked again of this workflow.

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
| `ICAE` | `icae.smk`, `exps/2026-06-21-icae-benchmark/` | Mistral-7B, 128 memory slots of dim 4096. Third-party weights and source tree: set `icae_weights`, `icae_code_dir`, `icae_base_model` |
| `T2L` | `t2l.smk`, `exps/2026-09-18-t2l-baseline/` | Text-to-LoRA: a hypernetwork that expands a frozen `gte-large` vector into a LoRA. Separate clone, like `doc-to-lora`: set `t2l_src` |
| `ActPatch` | `exps/2026-09-18-gene-vs-activation/` | training-free activation patching of Qwen3-4B hidden states. Numbers are typed into App. actpatch; run the scripts directly |
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
8. **Hardcoded absolute paths removed.** Several `exps/` scripts carried
   `/home/skojaku/projects/doc2lora-embedding/...` on `sys.path`; they now resolve
   the repository root from `__file__`.

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
    `exps/2026-07-02-simplex-kwgrid/icae_slots.py` here: a thin wrapper over
    `icae_lib.load_icae` that resolves the third-party weights, source tree, and base
    model through `workflow/config.yaml`, so the ICAE arm of the edge decode has one
    dependency instead of a chain of sideways imports.
12. **Every cache a figure reads is tracked.** Fig. cluster-labels draws from six
    JSON/CSV summaries of GPU decodes and judge panels. They are small, they are the
    published numbers, and they are committed with `.gitignore` exceptions, so the
    figure redraws on a laptop from a fresh clone. The expensive runs behind them
    stay reproducible through their own rules.

Scripts that no rule reaches (`eval_next_paper.py`, `eval_yearmean.py`,
`year_balanced_collab.py`, `specter2_embed.py`) are kept for provenance and now say
so in their own docstrings, so they cannot be mistaken for pipeline steps.

## What was left out, and why

This repository carries 23 rule files. The chains that are absent produced results
that never reached the manuscript: simplex fusion and the fusability sweeps,
topological (TDA) hole-finding, the idea-gene GA, cross-domain analogy, arXiv field
splits, interpolation demos, and the multi-agent holes study.

## Cold-start job counts

From nothing but the corpora and checkpoints (`snakemake -n <target>`):

| Target | Jobs | Heaviest step |
|---|---|---|
| `paper` / `paper_assets` | 195 | 3 × 644k-paper gene extraction (GPU-days) |
| `uncertainty` | 133 | 1000× bootstrap over every paired score pool |
| `general` | 115 | the general adapter + applying it everywhere |
| `s2and` | 78 | gene extraction for five disambiguation corpora |
| `kron` | 55 | per-field citation adapters |
| `temporal_hardening` | 52 | retraining the transform on pre-2018 citations |
| `figure2` | 34 | the pair-axis edge decodes behind panels (e), (f) |
| `fields` | 19 | economics + psychology corpora and genes |
| `label_eval` | 18 | LLM judge panel (OpenRouter, costs money) |
| `pair_axis` | 13 | 250 pairs × 13 points × 2 decoders |
| `efficiency` | 11 | throughput + ANN index measurements on GPU |
| `t2l` | 9 | the hypernetwork baseline's label and fusion arms |
| `baselines_aps` | 5 | SPECTER2 + INSTRUCTOR + GTE over 644k APS abstracts |
| `pacs_groups` | 5 | the PACS node set |

Every one of these was checked with `snakemake -n` against an **empty** `data/`
directory, so the counts are true cold-start figures: the only inputs assumed to
exist are the licensed corpora and the checkpoints named in
`workflow/config.yaml`.

With the `results` artifact bundle unpacked, the table-building tail of
`paper_assets` runs on a CPU in minutes.

Fig. cluster-labels is cheaper than the `figure2` job count suggests: it redraws in
seconds from the caches tracked in git — see the panel-by-panel table above for the
two commands that bypass the workflow's GPU producers.
