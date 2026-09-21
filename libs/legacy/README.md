# doc2lora-legacy

A frozen, verbatim copy of the **full** original `doc2lora` API, kept so the research
sandboxes under `exps/` keep working unchanged. It is imported as `doc2lora_legacy`.

```bash
pip install -e libs/legacy
```

**Do not use this for new code.** The maintained, minimal package is
[`doc2lora`](../doc2lora) — it exposes one clean representation path plus an
ergonomic `decode_gene` helper. This legacy package additionally carries the
perceiver encoder-latent path (`internalize_from_latents`, `extract_encoder_latents`)
and the niche multi-document arithmetic (`mix_query_vectors`, `splice_query_vectors`,
`cluster_query_vectors`, `stack_doc_pooled_vectors`, `even_split_counts`) that only the
`exps/` sandboxes rely on.
