#!/usr/bin/env bash
# Rust vs Python throughput benchmark. Usage: scripts/bench.sh [threads...]
set -e
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$HF_PROCESSORS_VENV/bin/activate"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
cargo build --release -q --example bench --features rayon,pil-jpeg
for t in ${@:-1 4}; do
  echo "### Rust (threads=$t)"; $CARGO_TARGET_DIR/release/examples/bench $t
  echo "### Python (threads=$t)"; OMP_NUM_THREADS=$t python golden/bench.py $t 2>/dev/null
done
uptime
