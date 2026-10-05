#!/usr/bin/env bash
# Run python from the project venv inside WSL. Usage: scripts/py.sh golden/make_golden.py
set -e
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$HF_PROCESSORS_VENV/bin/activate"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec python "$@"
