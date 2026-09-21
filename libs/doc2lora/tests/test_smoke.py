"""Import smoke tests — the package must import without the backbone/checkpoint."""


def test_import_and_version():
    import doc2lora

    assert isinstance(doc2lora.__version__, str)
    assert doc2lora.__version__


def test_public_api_symbols_present():
    import doc2lora

    for name in (
        "load_model",
        "resolve_checkpoint",
        "extract_norm_lora_emb",
        "mix_embeddings",
        "decode_adapter",
        "summarize_cluster",
    ):
        assert hasattr(doc2lora, name), name


def test_device_resolution_cpu():
    from doc2lora.device import resolve_device

    assert resolve_device("cpu").type == "cpu"
    # auto never raises and returns a concrete device
    assert resolve_device(None).type in ("cpu", "cuda", "mps")
