#!/usr/bin/env bash
# Build the abi3 Python wheel (bindings/python) and install it into the project venv.
# Usage: scripts/build_wheel.sh [extra maturin args, e.g. --target x86_64-pc-windows-gnu]
set -e
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$HF_PROCESSORS_VENV/bin/activate"
cd "$(dirname "${BASH_SOURCE[0]}")/../bindings/python"
OUT=$CARGO_TARGET_DIR/wheels
maturin build --release --out "$OUT" "$@"
if [ $# -eq 0 ]; then
  uv pip install --force-reinstall --no-deps "$(ls -t "$OUT"/hf_processors_rs-*-linux_x86_64.whl "$OUT"/hf_processors_rs-*manylinux*.whl 2>/dev/null | head -1)"
fi
ls -la "$OUT"
