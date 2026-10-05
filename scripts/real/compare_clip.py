"""Compare the clip-candle example (hf-processors + candle) with transformers on real photos.

    python scripts/real/compare_clip.py MODEL_DIR IMAGE_DIR RUST_OUT_DIR

MODEL_DIR is a local copy of openai/clip-vit-base-patch32 (weights, config, tokenizer files),
RUST_OUT_DIR the output of `examples/clip-candle`. transformers runs `CLIPProcessor` (the
default torchvision-backed `CLIPImageProcessor`) and `CLIPModel.get_image_features` on CPU
torch, on PIL images opened like `Image.open(path)`. Prints a JSON summary.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def image_features(model, pv):
    out = model.get_image_features(pixel_values=pv)
    # transformers v5 may return a model output instead of a tensor.
    return out if isinstance(out, torch.Tensor) else out.pooler_output


def cosine(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a * b).sum(1)


def main():
    model_dir, image_dir, rust_dir = map(Path, sys.argv[1:4])
    files = (rust_dir / "files.txt").read_text().split()
    pv_rust = np.load(rust_dir / "pixel_values.npy")
    emb_rust = np.load(rust_dir / "image_embeds.npy")
    torch.set_num_threads(4)
    proc = CLIPProcessor.from_pretrained(model_dir)
    model = CLIPModel.from_pretrained(model_dir).eval()

    pvs, embs, embs_on_rust_pv = [], [], []
    t = time.time()
    with torch.no_grad():
        for i in range(0, len(files), 16):
            images = [Image.open(image_dir / f) for f in files[i:i + 16]]
            pv = proc(images=images, return_tensors="pt")["pixel_values"]
            pvs.append(pv.numpy())
            embs.append(image_features(model, pv).numpy())
            embs_on_rust_pv.append(image_features(model, torch.from_numpy(pv_rust[i:i + 16])).numpy())
    pv_py, emb_py, emb_py_rpv = np.concatenate(pvs), np.concatenate(embs), np.concatenate(embs_on_rust_pv)
    modes = {}
    for f in files:
        with Image.open(image_dir / f) as im:
            modes[im.mode] = modes.get(im.mode, 0) + 1

    same_pv = [bool(np.array_equal(pv_py[i], pv_rust[i])) for i in range(len(files))]
    cos = cosine(emb_rust.astype(np.float64), emb_py.astype(np.float64))
    summary = {
        "images": len(files),
        "image_modes": modes,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "image_processor": type(proc.image_processor).__name__,
        "pixel_values_identical_images": int(sum(same_pv)),
        "pixel_values_max_abs_diff": float(np.abs(pv_py - pv_rust).max()),
        "pixel_values_bytes_identical": bool(pv_py.tobytes() == pv_rust.astype(np.float32).tobytes()),
        "embeds_max_abs_diff": float(np.abs(emb_py - emb_rust).max()),
        "embeds_max_abs": float(np.abs(emb_py).max()),
        "embeds_cosine_min": float(cos.min()),
        "embeds_cosine_mean": float(cos.mean()),
        # Model numerics alone: torch on the Rust pixel_values vs torch on its own.
        "torch_on_rust_pixel_values_max_abs_diff": float(np.abs(emb_py_rpv - emb_py).max()),
        "seconds_transformers": round(time.time() - t, 1),
    }
    print(json.dumps(summary, indent=1))
    mismatched = [f for f, ok in zip(files, same_pv) if not ok]
    if mismatched:
        print("pixel_values differ for:", mismatched[:20])


if __name__ == "__main__":
    main()
