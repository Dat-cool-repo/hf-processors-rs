#!/usr/bin/env bash
# Create the Python venv used to generate golden fixtures and run the Python benchmark.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Requires uv (https://docs.astral.sh/uv/).
[ -d "$HF_PROCESSORS_VENV" ] || uv venv "$HF_PROCESSORS_VENV" --python 3.12
source "$HF_PROCESSORS_VENV/bin/activate"
# Versions the golden fixtures were generated with (see golden/manifest.json).
# CPU-only torch: install from the PyTorch CPU index only.
uv pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchvision==0.29.1
uv pip install transformers==5.18.0 pillow==12.3.0 numpy==2.5.3 maturin pytest
python -c "import transformers, PIL, numpy, torch, torchvision; print(transformers.__version__, PIL.__version__, numpy.__version__, torch.__version__, torchvision.__version__)"
