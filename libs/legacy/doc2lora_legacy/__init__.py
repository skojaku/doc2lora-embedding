from doc2lora_legacy.model import load_model
from doc2lora_legacy.embed import (
    extract_encoder_latents,
    extract_encoder_latents_batch,
    extract_norm_lora_emb_batch,
    mean_pool_and_flatten,
)
from doc2lora_legacy.generate import internalize_from_latents, generate_text
from doc2lora_legacy.arithmetic import (
    extract_norm_lora_emb,
    mix_embeddings,
    mean_embeddings,
    interpolate_embeddings,
    splice_query_vectors,
    even_split_counts,
    mix_query_vectors,
    cluster_query_vectors,
    stack_doc_pooled_vectors,
    internalize_from_norm_lora_emb,
)
from doc2lora_legacy.cluster_summary import (
    cluster_centroid,
    pooled_to_norm_lora_emb,
    decode_cluster,
    summarize_cluster,
    DEFAULT_LABEL_PROMPT,
    FIELD_LABEL_PROMPT,
)

__all__ = [
    "load_model",
    "extract_encoder_latents",
    "extract_encoder_latents_batch",
    "extract_norm_lora_emb_batch",
    "mean_pool_and_flatten",
    "internalize_from_latents",
    "generate_text",
    "extract_norm_lora_emb",
    "mix_embeddings",
    "mean_embeddings",
    "interpolate_embeddings",
    "splice_query_vectors",
    "even_split_counts",
    "mix_query_vectors",
    "cluster_query_vectors",
    "stack_doc_pooled_vectors",
    "internalize_from_norm_lora_emb",
    "cluster_centroid",
    "pooled_to_norm_lora_emb",
    "decode_cluster",
    "summarize_cluster",
    "DEFAULT_LABEL_PROMPT",
    "FIELD_LABEL_PROMPT",
]
