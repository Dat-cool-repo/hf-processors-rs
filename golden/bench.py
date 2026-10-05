#!/usr/bin/env python3
"""Python side of the throughput benchmark (mirrors examples/bench.rs).

    python golden/bench.py [threads]

Images are decoded once up front (PIL), so only the processor call is timed.
"""

import json
import statistics
import sys
import time
from pathlib import Path

import torch
import transformers
from PIL import Image

ROOT = Path(__file__).resolve().parent
threads = int(sys.argv[1]) if len(sys.argv) > 1 else 1
torch.set_num_threads(threads)


def batch(names, n):
    imgs = []
    for name in names:
        im = Image.open(ROOT / "images" / name)
        im.load()
        imgs.append(im)
    return [imgs[i % len(imgs)] for i in range(n)]


def time_it(proc, images, iters):
    proc(images, return_tensors="np")  # warm-up
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        proc(images, return_tensors="np")
        times.append(time.perf_counter() - t)
    return statistics.median(times)


typical = batch(["photo_640x480.png", "coffee.png", "astronaut.png", "hifreq_333x517.png", "photo_500x375.jpg"], 64)
large = batch(["huge_3000x2000.png"], 8)
configs = [
    ("clip-vit-base-patch32", "openai_clip-vit-base-patch32.json", "CLIPImageProcessor"),
    ("vit-base-patch16-224", "google_vit-base-patch16-224.json", "ViTImageProcessor"),
    ("siglip-so400m-patch14-384", "google_siglip-so400m-patch14-384.json", "SiglipImageProcessor"),
]
print(f"threads={threads} transformers={transformers.__version__} torch={torch.__version__}")
print(f"{'config':<28} {'backend':<12} {'batch':<26} {'ms/batch':>10} {'images/s':>12}")
for name, cfg_file, cls in configs:
    cfg = json.loads((ROOT / "configs" / cfg_file).read_text())
    for backend, suffix in [("Torchvision", ""), ("Pil", "Pil")]:
        proc = getattr(transformers, cls + suffix).from_dict(dict(cfg))
        for label, imgs, iters in [("64 x ~0.3MP (typical)", typical, 15), ("8 x 6MP (3000x2000)", large, 5)]:
            t = time_it(proc, imgs, iters)
            print(f"{name:<28} {backend:<12} {label:<26} {t * 1e3:>10.1f} {len(imgs) / t:>12.0f}", flush=True)
