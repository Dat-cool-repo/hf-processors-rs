#!/usr/bin/env bash
set -e
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
cargo build --release -q --example bench --features rayon,pil-jpeg
for t in ${@:-1}; do $CARGO_TARGET_DIR/release/examples/bench $t; done
