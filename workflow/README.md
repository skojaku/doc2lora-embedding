Rules and the scripts they call.

- `rules/*.smk` — one file per chain. A rule file is named for what it produces,
  and its header comment says which figure, table, or claim that is.
- `scripts/` — the steps the rules invoke. `bench_data.py` is the single place
  that answers where a corpus lives (`$ENV_VAR` → `workflow/config.yaml` →
  built-in default); `text_encoders.py` holds one `build_*` per text baseline
  behind a shared `encode(texts)`.
- `plot/` — figure and table generators. Every one is dual-mode: Snakemake
  passes paths through the `snakemake` object, and the same file runs from a
  shell with explicit arguments.
- `config.template.yaml` — copy to `config.yaml` and edit. That copy is
  gitignored, so machine paths never land in git.

See `REPRODUCE.md` for the result → code map.
