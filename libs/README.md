# libs/

Installable Python packages used across the project.

- **[`doc2lora/`](doc2lora/)** — the public, reusable package: turn documents
  into compact adapters, do arithmetic on them, and decode them back into text.
  This is the thing you'd `pip install`. See [its README](doc2lora/README.md).
- **`legacy/`** — the frozen `doc2lora_legacy` API that the older workflow
  scripts import. Kept so those chains reproduce unchanged; not for new work.
