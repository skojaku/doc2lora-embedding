# doc2lora

Turn a document into a compact **adapter** — a small LoRA-shaped embedding that you can
do *arithmetic* on (average, interpolate, blend) and **decode back into text**. Blend two
papers and read out the fused idea; label a cluster straight from its members' adapters.

```python
from doc2lora import load_model, extract_norm_lora_emb, decode_adapter, interpolate_embeddings

# load_model returns (model, gen_tokenizer, ctx_tokenizer)
model, gen_tok, ctx_tok = load_model(CKPT)

game  = extract_norm_lora_emb(model, ctx_tok, "Game theory models strategic interaction among rational agents...")
evol  = extract_norm_lora_emb(model, ctx_tok, "Evolutionary biology explains how populations change via mutation and selection...")

# decode an adapter back to text
print(decode_adapter(model, gen_tok, game))      # -> "...game theory / strategic decision-making..."

# fuse two ideas and read out the blend
blend = interpolate_embeddings(game, evol, alpha=0.5)
print(decode_adapter(model, gen_tok, blend))     # -> "...evolutionary game theory..."
```

## Install

```bash
pip install -e libs/doc2lora            # core
pip install -e libs/doc2lora[dev]       # + pytest, ruff (to run the tests)
```

`import doc2lora` and all the pure tensor ops (mix / mean / interpolate /
cluster centroids) work with just the core install — **no GPU and no
checkpoint** — so the library runs on CPU/MPS too (slower, but it works).

The heavy half — `load_model`, `internalize`, `decode` — needs the model
backbone (`ModulatedPretrainedModel`, the tokenizers, the LoRA merge/apply
ops), which lives in **`ctx_to_lora`**. That package is *not* on PyPI, so
install it as a second step (and, on CUDA, build the `flash-attn` wheel — see
the root README):

```bash
git clone https://github.com/SakanaAI/doc-to-lora
pip install -e doc-to-lora               # provides the `ctx_to_lora` package
pip install -e libs/doc2lora[gpu]        # declares flash-attn (needs a build toolchain)
```

If the backbone is missing, `load_model` raises a clear install message rather
than a bare `ImportError`.

### Checkpoints

A `CKPT` is a doc2lora checkpoint `.bin` (the released ones are on Hugging Face at
[SakanaAI/doc-to-lora](https://huggingface.co/SakanaAI/doc-to-lora)). `load_model()`
resolves it in one canonical order: the explicit argument, then the
`$DOC2LORA_CKPT` environment variable, then a repo-relative fallback under
`data/agent_assets/<model>/checkpoint-<step>/pytorch_model.bin` (the lightweight
one is `data/agent_assets/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin`). So
`load_model()` with no argument works once `$DOC2LORA_CKPT` is set or a
checkpoint sits under `data/agent_assets/`.

## The API, in one screen

```python
from doc2lora import (
    load_model,                       # load checkpoint + both tokenizers

    # document -> adapter
    extract_norm_lora_emb,            # one document
    extract_norm_lora_emb_batch,      # many (length-sorted batching)
    mean_pool_and_flatten,            # adapter -> flat 1-D vector (storage / similarity)

    # arithmetic on adapters (all return an adapter of the same shape)
    interpolate_embeddings,           # (1-a)*x + a*y  — fuse / extrapolate two adapters
    mean_embeddings,                  # uniform mean of several adapters
    mix_embeddings,                   # weighted (barycentric) mix of several adapters
    internalize_from_norm_lora_emb,   # inject an adapter into the model

    # adapter -> text
    decode_adapter,                   # one-call: internalize -> generate -> reset
    chat_with_adapter,                # multi-turn chat conditioned on an adapter
    generate_text,                    # raw generation after you internalize yourself

    # label a cluster of documents from their adapters
    summarize_cluster, cluster_centroid, decode_cluster,
    DEFAULT_LABEL_PROMPT, FIELD_LABEL_PROMPT,
)
```

An **adapter** (`norm_lora_emb`) has shape `[n_layers, n_modules, r, latent]` (e.g.
`[36, 1, 8, 512]`); each `[..., latent]` slice is L2-normalised. The head that turns an
adapter into the model's LoRA weights is *linear*, so a weighted average of adapters is
exactly the same weighted average of the weights they produce — that is why adapter
arithmetic decodes to sensible fused ideas.

`mean_pool_and_flatten` pools an adapter over the module/rank axes into a flat
`[n_layers * latent]` vector — convenient for storing many adapters on disk and for cosine
similarity.

## Cluster labeling

Decode a label for a *cluster* of documents directly from their pooled adapters — no
full-rank extraction needed.

```python
import numpy as np
from doc2lora import load_model, summarize_cluster, FIELD_LABEL_PROMPT

model, gen_tok, ctx_tok = load_model(CKPT)

adapters = np.load("qwen_norm_lora_emb.npz")["embeddings"][member_rows]  # (N, n_layers*latent)

label = summarize_cluster(model, gen_tok, adapters, n_layers=36,
                          op="intersection", prompt=FIELD_LABEL_PROMPT)
```

Recipe (the defaults, validated on APS/PACS): aggregate members into a
`[n_layers, latent]` centroid (`op="mean"` for narrow clusters, `op="intersection"` for
broad ones), expand across the `r` rank slots **without renormalizing**, and decode with a
single constrained prompt whose wording sets the abstraction altitude (`FIELD_LABEL_PROMPT`
for fields, `DEFAULT_LABEL_PROMPT` for topics). See `cluster_summary.py` for the full
rationale and gotchas.

## Notes

- The model holds two copies (a context encoder and a generator). `load_model(CKPT,
  mode="embed")` keeps only the encoder, `mode="generate"` only the generator — handy when
  GPU memory is tight and you do embedding and decoding in separate phases.
- The helpers call `model.reset()` for you between adapters; if you internalize manually
  with `internalize_from_norm_lora_emb`, remember to `model.reset()` before the next one.
- Need the older, broader API (perceiver encoder-latent path, multi-document
  query-vector ops)? It is preserved verbatim as the `doc2lora_legacy` package in
  `libs/legacy/`.
