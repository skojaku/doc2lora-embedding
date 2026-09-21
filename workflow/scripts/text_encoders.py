"""Shared registry of scientific-text embedding BASELINE encoders.

Each encoder is exposed via ``get_encoder(name)`` and returns a callable

    encode(texts: list[str]) -> np.ndarray  # [N, D] float32

Encoders are batched, use the GPU when available, and emit fp32. Vectors are
returned UN-normalized (the caller / scorer L2-normalizes for cosine).

SPECTER2 also exposes ``encode_pairs(titles, abstracts)`` because its native
input convention is ``title + sep_token + abstract``.

Importable AND runnable:
    python text_encoders.py --smoke            # smoke-test every encoder
    python text_encoders.py --smoke --methods specter2 instructor

Registry: REGISTRY maps method name -> builder. Methods:
    sbert      sentence-transformers/all-mpnet-base-v2          (768)
    gte_large  Alibaba-NLP/gte-large-en-v1.5 (CLS)              (1024)  <- T2L's input
    specter2   allenai/specter2_base (+ proximity adapter)      (768)
    instructor hkunlp/instructor-large                          (768)
    text2vec   shibing624/text2vec-base-multilingual            (384)

SPECTER2 adapter note
---------------------
The `adapters` library (AutoAdapterModel + load_adapter("allenai/specter2"))
is the *intended* path, but `adapters>=1.3` requires transformers~=4.57 while
this env is pinned to transformers 4.51.3 (required by doc2lora / ctx-to-lora).
We therefore try the adapter path first and FALL BACK to a raw
`transformers.AutoModel` load of `allenai/specter2_base` (the base SciBERT-init
encoder, CLS-pooled) when the adapter path raises. Both yield 768-dim CLS
embeddings; the adapter adds the proximity task head trained for retrieval.
``build_specter2`` records which path was used in ``encoder.adapter_path``.
"""
import argparse
import sys

import numpy as np


# --------------------------------------------------------------------------- #
# device helper
# --------------------------------------------------------------------------- #
def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# SBERT  (all-mpnet-base-v2)
# --------------------------------------------------------------------------- #
def build_sbert(model_name="sentence-transformers/all-mpnet-base-v2", batch_size=256, gpu=True):
    from sentence_transformers import SentenceTransformer

    dev = _device() if gpu else "cpu"
    model = SentenceTransformer(model_name, device=dev)

    def encode(texts):
        v = model.encode(list(texts), batch_size=batch_size, convert_to_numpy=True,
                         show_progress_bar=False, normalize_embeddings=False)
        return np.asarray(v, dtype=np.float32)

    encode.dim = model.get_sentence_embedding_dimension()
    encode.name = "sbert"
    return encode


# --------------------------------------------------------------------------- #
# SPECTER2  (allenai/specter2_base + proximity adapter; raw-AutoModel fallback)
# --------------------------------------------------------------------------- #
def build_specter2(base="allenai/specter2_base", adapter="allenai/specter2",
                   batch_size=64, gpu=True, max_length=512):
    import torch
    from transformers import AutoTokenizer

    dev = _device() if gpu else "cpu"
    tok = AutoTokenizer.from_pretrained(base)

    adapter_path = None
    model = None
    # --- intended path: adapters lib ---
    try:
        from adapters import AutoAdapterModel
        model = AutoAdapterModel.from_pretrained(base)
        model.load_adapter(adapter, source="hf", load_as="proximity", set_active=True)
        adapter_path = "adapters"
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[specter2] adapter path unavailable ({type(e).__name__}: "
                         f"{str(e)[:140]}); falling back to raw AutoModel base.\n")
        from transformers import AutoModel
        model = AutoModel.from_pretrained(base)
        adapter_path = "raw_automodel_base"

    model = model.to(dev).eval()

    def _encode_inputs(text_inputs):
        out = []
        with torch.no_grad():
            for s in range(0, len(text_inputs), batch_size):
                batch = text_inputs[s:s + batch_size]
                enc = tok(batch, padding=True, truncation=True, max_length=max_length,
                          return_tensors="pt").to(dev)
                res = model(**enc)
                cls = res.last_hidden_state[:, 0, :]  # CLS token
                out.append(cls.cpu().float().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def encode_pairs(titles, abstracts):
        sep = tok.sep_token or "[SEP]"
        inputs = [f"{(t or '')}{sep}{(a or '')}" for t, a in zip(titles, abstracts)]
        return _encode_inputs(inputs)

    # plain-text fallback: callers that pass a single string get it embedded as-is
    def encode(texts):
        return _encode_inputs(list(texts))

    encode.encode_pairs = encode_pairs
    encode.adapter_path = adapter_path
    encode.dim = model.config.hidden_size
    encode.name = "specter2"
    return encode


# --------------------------------------------------------------------------- #
# INSTRUCTOR  (hkunlp/instructor-large)
# --------------------------------------------------------------------------- #
INSTRUCTOR_INSTRUCTION = "Represent the scientific paper abstract for retrieval:"


def build_instructor(model_name="hkunlp/instructor-large", batch_size=64, gpu=True,
                     instruction=INSTRUCTOR_INSTRUCTION):
    from sentence_transformers import SentenceTransformer

    dev = _device() if gpu else "cpu"
    model = SentenceTransformer(model_name, device=dev)

    def encode(texts):
        pairs = [[instruction, t] for t in texts]
        v = model.encode(pairs, batch_size=batch_size, convert_to_numpy=True,
                         show_progress_bar=False, normalize_embeddings=False)
        return np.asarray(v, dtype=np.float32)

    encode.dim = model.get_sentence_embedding_dimension()
    encode.name = "instructor"
    return encode


# --------------------------------------------------------------------------- #
# text2vec  (shibing624/text2vec-base-multilingual)
# --------------------------------------------------------------------------- #
def build_text2vec(model_name="shibing624/text2vec-base-multilingual", batch_size=64, gpu=True):
    dev = _device() if gpu else "cpu"
    try:
        from text2vec import SentenceModel
        model = SentenceModel(model_name, device=dev)

        def encode(texts):
            v = model.encode(list(texts), batch_size=batch_size, show_progress_bar=False,
                             convert_to_numpy=True, normalize_embeddings=False)
            return np.asarray(v, dtype=np.float32)

        encode.dim = getattr(model, "get_sentence_embedding_dimension", lambda: None)() \
            or model.bert.config.hidden_size
    except Exception:  # noqa: BLE001 — fall back to plain sentence-transformers load
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device=dev)

        def encode(texts):
            v = model.encode(list(texts), batch_size=batch_size, convert_to_numpy=True,
                             show_progress_bar=False, normalize_embeddings=False)
            return np.asarray(v, dtype=np.float32)

        encode.dim = model.get_sentence_embedding_dimension()

    encode.name = "text2vec"
    return encode


# --------------------------------------------------------------------------- #
# EmbeddingGemma  (google/embeddinggemma-300m)
# --------------------------------------------------------------------------- #
# Gemma3-based sentence embedder with task-specific prompts. We embed paper text
# as a DOCUMENT corpus (symmetric paper<->paper similarity), so the "document"
# prompt ("title: none | text: ...") is applied to every text — the model-native
# analogue of INSTRUCTOR's instruction.
#
# IMPORTANT: EmbeddingGemma needs transformers>=4.56 for `use_bidirectional_attention`;
# under the repo's pinned 4.51.3 it SILENTLY falls back to causal attention (degraded
# embeddings). Run this encoder in the isolated `.venv-emgemma` (newer transformers),
# not the main env. The builder asserts the parameter is honored unless EMGEMMA_ALLOW_CAUSAL=1.
def build_embeddinggemma(model_name="google/embeddinggemma-300m", batch_size=256, gpu=True,
                         prompt_name="document", max_seq_length=512):
    import os
    import transformers
    from packaging.version import parse as vparse
    from sentence_transformers import SentenceTransformer

    if vparse(transformers.__version__) < vparse("4.56") and not os.environ.get("EMGEMMA_ALLOW_CAUSAL"):
        raise RuntimeError(
            f"EmbeddingGemma needs transformers>=4.56 for bidirectional attention; found "
            f"{transformers.__version__}. Run in .venv-emgemma, or set EMGEMMA_ALLOW_CAUSAL=1 "
            f"to accept degraded (causal) embeddings.")

    dev = _device() if gpu else "cpu"
    model = SentenceTransformer(model_name, device=dev)
    # Cap sequence length: the model's 2048 default + attention's O(L^2) cost makes long
    # docs explode VRAM. Abstracts fit easily in 512 (field corpora p99 ~745 tok); truncate.
    if max_seq_length:
        model.max_seq_length = int(max_seq_length)
    avail = set(getattr(model, "prompts", {}) or {})
    pn = prompt_name if prompt_name in avail else None
    if prompt_name and pn is None:
        sys.stderr.write(f"[embeddinggemma] prompt {prompt_name!r} not in model prompts "
                         f"{sorted(avail)}; embedding without a prompt.\n")

    def encode(texts):
        kw = dict(batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False,
                  normalize_embeddings=False)
        if pn is not None:
            kw["prompt_name"] = pn
        v = model.encode(list(texts), **kw)
        return np.asarray(v, dtype=np.float32)

    try:
        encode.dim = model.get_sentence_embedding_dimension()
    except Exception:  # noqa: BLE001 — method renamed in sentence-transformers 5.x
        encode.dim = model.get_embedding_dimension()
    encode.name = "embeddinggemma"
    encode.prompt_name = pn
    return encode


# --------------------------------------------------------------------------- #
# gte-large-en-v1.5  --  Text-to-LoRA's native coordinates (#151)
# --------------------------------------------------------------------------- #
def build_gte_large(model_name="Alibaba-NLP/gte-large-en-v1.5", batch_size=64,
                    gpu=True, max_seq_length=512):
    """CLS-pooled gte-large-en-v1.5 (1,024-d).

    This is NOT a new text baseline added on its own merits -- decision D6 of #142 closed
    #69 by hedging the uniqueness claims rather than adding 2024+ encoders, and bge/gte/
    e5mistral stay OFF in `workflow/config.yaml:gcb_new_encoders`. It is here because
    gte-large IS Text-to-LoRA's embedding: its hypernetwork reads a frozen gte vector and
    expands it into an adapter, so the honest retrieval row for the T2L baseline is
    labelled "Text-to-LoRA's native coordinates = gte-large", not presented as an adapter
    space (#151).

    Pooling is CLS, matching `hyper_llm_modulator.utils.model_loading`'s gte branch, and
    the model is loaded in fp32 there too -- a different pooling would not be T2L's input.
    Needs `trust_remote_code=True` (Alibaba-NLP/new-impl).
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = _device() if gpu else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32,
                                      trust_remote_code=True).to(dev).eval()

    @torch.no_grad()
    def encode(texts):
        out = []
        texts = list(texts)
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], padding=True, truncation=True,
                      max_length=max_seq_length, return_tensors="pt").to(dev)
            h = model(**enc).last_hidden_state[:, 0]        # CLS, as T2L pools it
            out.append(h.float().cpu().numpy())
        return np.concatenate(out, 0).astype(np.float32)

    encode.dim = model.config.hidden_size
    encode.name = "gte_large"
# GTE  (Alibaba-NLP/gte-large-en-v1.5)
# --------------------------------------------------------------------------- #
def build_gte(model_name="Alibaba-NLP/gte-large-en-v1.5", batch_size=64, gpu=True):
    from sentence_transformers import SentenceTransformer

    dev = _device() if gpu else "cpu"
    model = SentenceTransformer(model_name, device=dev, trust_remote_code=True)

    def encode(texts):
        v = model.encode(list(texts), batch_size=batch_size, convert_to_numpy=True,
                         show_progress_bar=False, normalize_embeddings=False)
        return np.asarray(v, dtype=np.float32)

    encode.dim = model.get_sentence_embedding_dimension()
    encode.name = "gte"
    return encode


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
REGISTRY = {
    "sbert": build_sbert,
    "specter2": build_specter2,
    "instructor": build_instructor,
    "text2vec": build_text2vec,
    "embeddinggemma": build_embeddinggemma,   # document prompt (see build_embeddinggemma default)
    "gte": build_gte,                         # similarity-benchmark baseline
    "gte_large": build_gte_large,             # Text-to-LoRA's native coordinates (#151)
}

# methods whose native input is (title, abstract) rather than a single text
PAIR_INPUT_METHODS = {"specter2"}


def get_encoder(name, **kwargs):
    if name not in REGISTRY:
        raise KeyError(f"unknown encoder {name!r}; choices: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)


# --------------------------------------------------------------------------- #
# smoke test
# --------------------------------------------------------------------------- #
def _smoke(methods, gpu):
    titles = [
        "Graph neural networks for molecules",
        "Message passing on molecular graphs",
        "Superconductivity in cuprates",
        "High-Tc superconductors phase diagram",
        "Auction theory and mechanism design",
        "Nash equilibrium in repeated games",
        "Quantum error correction codes",
        "Stabilizer codes for fault tolerance",
    ]
    abstracts = [
        "We learn molecular properties with graph neural networks over atom bonds.",
        "A message-passing scheme aggregates neighbor atoms to predict properties.",
        "We study electron pairing and the critical temperature of copper-oxide superconductors.",
        "The phase diagram of high-temperature cuprate superconductors is mapped versus doping.",
        "We design truthful auction mechanisms maximizing revenue under incentive constraints.",
        "Folk theorems characterize Nash equilibria sustained in infinitely repeated games.",
        "Quantum codes protect logical qubits from decoherence via syndrome measurement.",
        "Stabilizer formalism yields fault-tolerant codes with transversal logical gates.",
    ]
    texts = [f"{t}. {a}" for t, a in zip(titles, abstracts)]

    def cos3(V):
        Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        S = Vn @ Vn.T
        return S[:3, :3]

    results = {}
    for m in methods:
        print(f"\n=== {m} ===")
        try:
            enc = get_encoder(m, gpu=gpu)
            if m in PAIR_INPUT_METHODS:
                V = enc.encode_pairs(titles, abstracts)
                print(f"  adapter_path: {getattr(enc, 'adapter_path', 'n/a')}")
            else:
                V = enc(texts)
            print(f"  OK  shape={V.shape}  dtype={V.dtype}  dim={getattr(enc, 'dim', '?')}")
            S = cos3(V)
            print("  3x3 cosine submatrix (rows 0,1=related GNN; row 2=superconductivity):")
            for r in S:
                print("   ", "  ".join(f"{x:+.3f}" for x in r))
            ok_diag = np.allclose(np.diag(S), 1.0, atol=1e-3)
            ok_rel = S[0, 1] > S[0, 2]  # GNN0~GNN1 should beat GNN0~superconductivity
            print(f"  sanity: diag~1 {ok_diag} | related>unrelated {ok_rel} ({S[0,1]:+.3f} > {S[0,2]:+.3f})")
            results[m] = {"ok": True, "shape": tuple(V.shape), "diag": bool(ok_diag),
                          "related_gt_unrelated": bool(ok_rel)}
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results[m] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print("\n=== SMOKE SUMMARY ===")
    for m, r in results.items():
        print(f"  {m:10s} {r}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--methods", nargs="+", default=list(REGISTRY), choices=list(REGISTRY))
    ap.add_argument("--no-gpu", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        _smoke(a.methods, gpu=not a.no_gpu)
    else:
        print("text_encoders registry:", list(REGISTRY))
        print("run with --smoke to test")
