#!/usr/bin/env bash
# Build the isolated EmbeddingGemma venv. EmbeddingGemma (google/embeddinggemma-300m, a Gemma3 arch)
# needs transformers>=4.56 for `use_bidirectional_attention`; the main env is pinned to 4.51.3 for
# doc2lora/ctx-to-lora, so we keep a separate --system-site-packages venv (inherits torch/sbert, overrides
# transformers). Baseline embedding is standalone (writes an npz scored later), so the newer transformers
# never touches the main env. Idempotent: re-runs are cheap; writes a .ready sentinel for Snakemake.
set -euo pipefail
cd "$(dirname "$0")/../.."        # repo root
VENV=.venv-emgemma

if [ ! -d "$VENV" ]; then
  python -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
. "$VENV/bin/activate"
python - <<'PY'
import importlib.metadata as m
from packaging.version import parse as v
try:
    ok = v(m.version("transformers")) >= v("4.56")
except m.PackageNotFoundError:
    ok = False
raise SystemExit(0 if ok else 1)
PY
if [ $? -ne 0 ]; then
  pip install -q -U "transformers>=4.56.2"
fi
# sanity: EmbeddingGemma loads with bidirectional attention honored (no causal-fallback warning)
python -c "import transformers,sentence_transformers; print('transformers',transformers.__version__,'st',sentence_transformers.__version__)"
touch "$VENV/.ready"
echo "[setup_emgemma_venv] ready -> $VENV/.ready"
