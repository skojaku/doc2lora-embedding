from importlib.metadata import PackageNotFoundError, version as _pkg_version

from doc2lora.model import load_model, resolve_checkpoint

try:
    __version__ = _pkg_version("doc2lora")
except PackageNotFoundError:  # not installed (e.g. run from a source checkout)
    __version__ = "0.2.0"
from doc2lora.embed import (
    extract_norm_lora_emb_batch,
    mean_pool_and_flatten,
)
from doc2lora.arithmetic import (
    extract_norm_lora_emb,
    mix_embeddings,
    mean_embeddings,
    interpolate_embeddings,
    internalize_from_norm_lora_emb,
)
from doc2lora.generate import generate_text, decode_adapter, chat_with_adapter
from doc2lora.bench_retrieval import (
    load_bench_emb,
    unit,
    cosine_scores,
    topk_indices,
    precision_at_ks,
)
from doc2lora.cluster_summary import (
    cluster_centroid,
    pooled_to_norm_lora_emb,
    decode_cluster,
    summarize_cluster,
    DEFAULT_LABEL_PROMPT,
    FIELD_LABEL_PROMPT,
)

__all__ = [
    "__version__",
    # load
    "load_model",
    "resolve_checkpoint",
    # embed: document -> adapter
    "extract_norm_lora_emb",
    "extract_norm_lora_emb_batch",
    "mean_pool_and_flatten",
    # arithmetic on adapters
    "mix_embeddings",
    "mean_embeddings",
    "interpolate_embeddings",
    "internalize_from_norm_lora_emb",
    # bench retrieval primitives
    "load_bench_emb",
    "unit",
    "cosine_scores",
    "topk_indices",
    "precision_at_ks",
    # decode: adapter -> text
    "generate_text",
    "decode_adapter",
    "chat_with_adapter",
    # cluster labeling
    "cluster_centroid",
    "pooled_to_norm_lora_emb",
    "decode_cluster",
    "summarize_cluster",
    "DEFAULT_LABEL_PROMPT",
    "FIELD_LABEL_PROMPT",
]
