"""Fetch openai/clip-vit-base-patch32 for the end-to-end check and write model.safetensors.

    python scripts/real/get_clip.py OUT_DIR

The repo's main branch ships the weights as pytorch_model.bin only (model.safetensors exists
only in open conversion PRs), so the official main-branch weights are converted locally:
every tensor is copied unchanged (contiguous, same dtype) into OUT_DIR/model.safetensors,
which both candle and transformers then load.
"""

import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

REPO = "openai/clip-vit-base-patch32"
FILES = ["pytorch_model.bin", "config.json", "preprocessor_config.json", "tokenizer_config.json",
         "tokenizer.json", "vocab.json", "merges.txt", "special_tokens_map.json"]

out = Path(sys.argv[1])
for f in FILES:
    print(hf_hub_download(REPO, f, local_dir=out), flush=True)
state = torch.load(out / "pytorch_model.bin", map_location="cpu", weights_only=True)
save_file({k: v.contiguous() for k, v in state.items()}, out / "model.safetensors", metadata={"format": "pt"})
(out / "pytorch_model.bin").unlink()  # transformers would prefer it; keep one copy of the weights
print(f"{len(state)} tensors -> {out / 'model.safetensors'}")
