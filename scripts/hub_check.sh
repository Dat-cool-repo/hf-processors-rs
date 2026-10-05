#!/usr/bin/env bash
# Network check: Rust from_pretrained(<hub repo>) vs Python AutoImageProcessor on the same image.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$HF_PROCESSORS_VENV/bin/activate"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
cargo build --release -q --example preprocess
for repo in openai/clip-vit-base-patch32 google/vit-base-patch16-224 google/siglip-base-patch16-224 facebook/convnext-tiny-224; do
  echo "== $repo"
  $CARGO_TARGET_DIR/release/examples/preprocess $repo golden/images/coffee.png
  python - "$repo" <<'PY' 2>/dev/null
import sys, numpy as np
from PIL import Image
from transformers import AutoImageProcessor
p = AutoImageProcessor.from_pretrained(sys.argv[1])
pv = p(Image.open("golden/images/coffee.png"), return_tensors="np")["pixel_values"][0]
print(f"python {type(p).__name__}: shape={list(pv.shape)} sum={pv.astype(np.float64).sum():.6f} first={pv.reshape(-1)[:4].tolist()}")
PY
done
