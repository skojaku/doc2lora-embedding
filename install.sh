#!/usr/bin/env bash
# Create the reproduction environment.
#
#   ./install.sh            # conda env "doc2lora" + the two local packages
#
# flash-attn is installed last and separately: the wheel must match
# torch 2.6.0+cu124 / cp310, and building it from source takes a while. It is
# only needed by the batched Qwen gene extractor; every CPU-side rule (scoring,
# bootstrap, tables) runs without it.
set -euo pipefail

ENV_NAME="${ENV_NAME:-doc2lora}"

conda env create -n "$ENV_NAME" -f environment.yml || conda env update -n "$ENV_NAME" -f environment.yml
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# The two local packages: the current API and the frozen legacy API the
# exps/ chains import.
pip install -e libs/doc2lora
pip install -e libs/legacy

# The hypernetwork implementation itself (ctx_to_lora) is a separate project.
# Clone it and point workflow/config.yaml:doc_to_lora_src at its src/.
echo
echo "Next:"
echo "  1. git clone <doc-to-lora> && set doc_to_lora_src in workflow/config.yaml"
echo "  2. pip install flash-attn==2.7.4.post1 --no-build-isolation   # GPU rules only"
echo "  3. cp workflow/config.template.yaml workflow/config.yaml && edit the paths"
