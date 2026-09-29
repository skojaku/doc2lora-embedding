#!/usr/bin/env bash
# Build the isolated BERTopic venv. bertopic pulls a numpy 2 / umap-learn / hdbscan stack that does not
# match the main env's pins (torch 2.6 + transformers 4.51.3 for doc2lora), and the BERTopic baseline is
# standalone -- it writes keyword-list JSONs that are scored later by the CPU eval -- so the newer stack
# never touches the main env. --system-site-packages lets it reuse torch / sentence-transformers / pyarrow.
# Idempotent: re-runs are cheap; writes a .ready sentinel for Snakemake.
set -euo pipefail
cd "$(dirname "$0")/../.."        # repo root
VENV=.venv-bertopic

if [ ! -d "$VENV" ]; then
  python -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
. "$VENV/bin/activate"
if ! python -c "import bertopic" 2>/dev/null; then
  pip install -q -U bertopic
fi
python -c "import bertopic, sklearn, numpy; print('bertopic', bertopic.__version__, '| sklearn', sklearn.__version__, '| numpy', numpy.__version__)"
touch "$VENV/.ready"
echo "[setup_bertopic_venv] ready -> $VENV/.ready"
