"""Device resolution helpers so the library runs on CPU / CUDA / MPS.

This is the *only* place a literal ``"cuda"`` should appear. Everything else
threads a resolved ``torch.device`` around, so a machine without an NVIDIA GPU
gets a working (if slower) CPU/MPS path instead of an opaque crash.
"""

import contextlib

import torch


def resolve_device(device=None):
    """Resolve a device argument to a concrete ``torch.device``.

    Args:
        device: ``None`` or ``"auto"`` auto-selects (CUDA if available, else
            Apple MPS, else CPU). Any explicit value (``"cpu"``, ``"cuda"``,
            ``"mps"``, an int, or a ``torch.device``) is passed through.

    Returns:
        torch.device
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def autocast_context(device, dtype=torch.bfloat16):
    """Return an autocast context appropriate for ``device``.

    CUDA and CPU support ``torch.autocast``; MPS (and anything else) does not
    reliably, so we fall back to a no-op context there rather than crashing.
    """
    dev = device if isinstance(device, torch.device) else torch.device(device)
    if dev.type in ("cuda", "cpu"):
        return torch.autocast(device_type=dev.type, dtype=dtype)
    return contextlib.nullcontext()
