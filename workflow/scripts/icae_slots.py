"""ICAE memory slots for the pair-axis decode: load the model, compress a document.

ICAE (Ge et al., 2024) compresses a document into 128 memory slots of dim 4096. The
fusion baseline averages those slot tensors and decodes the average, which is the
in-context-autoencoder analogue of averaging two adapters. `decode_absfollow_icae.py`
uses this module for the compression half and writes its own decoding loop, because
the pair-axis study asks for an abstract followed by keywords rather than the keyword
list the other ICAE entry points emit.

The weights, the ICAE source tree, and the base model are third-party and live
outside this repository; they resolve through `workflow/config.yaml`
(`icae_weights`, `icae_code_dir`, `icae_base_model`) or the matching environment
variables, the same way every other data path here does.

  ICAE_WEIGHTS=... ICAE_CODE_DIR=... python decode_absfollow_icae.py pairL1_00
"""
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "workflow" / "scripts"))
sys.path.insert(0, str(ROOT / "exps" / "2026-06-21-icae-benchmark"))
import bench_data                                        # noqa: E402
from icae_lib import load_icae                           # noqa: E402

DEVICE = os.environ.get("ICAE_DEVICE", "cuda")
MAX_LEN = 512

# Mistral-7B-Instruct's [INST] / [/INST] token ids: the memory slots are spliced
# between them, so the model reads them as the content of one instruction turn.
INST_LEFT = torch.LongTensor([[1, 733, 16289, 28793]]).to(DEVICE)
INST_RIGHT = [733, 28748, 16289, 28793]


def load_model():
    """The fine-tuned ICAE (Mistral-7B) in eval mode, on DEVICE."""
    return load_icae(bench_data.path("icae_weights"),
                     bench_data.path("icae_code_dir"),
                     bench_data.path("icae_base_model"),
                     device=DEVICE)


@torch.no_grad()
def compress(model, text):
    """One document -> its memory slots, [m, 4096] with m = 128 at MAX_LEN tokens."""
    ids = model.tokenizer(text, truncation=True, max_length=MAX_LEN, padding=False,
                          return_attention_mask=False)["input_ids"]
    slots = model._compress(torch.LongTensor([ids]).to(DEVICE))
    return slots.squeeze(0) if slots.dim() == 3 else slots
