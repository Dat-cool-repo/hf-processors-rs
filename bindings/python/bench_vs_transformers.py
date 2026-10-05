#!/usr/bin/env python3
"""Speed of hf_processors_rs vs transformers, called from Python (median of N runs).

    python bindings/python/bench_vs_transformers.py [threads ...]     (default: 1 4)

Inputs are PIL images decoded up front (the PIL -> numpy copy is included in our timings),
plus one "from files" row where both sides decode JPEG/PNG files themselves.
"""

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image

import hf_processors_rs as hpr

GOLDEN = Path(__file__).resolve().parents[2] / "golden"


def load(names, n):
    ims = []
    for name in names:
        im = Image.open(GOLDEN / "images" / name)
        im.load()
        ims.append(im)
    return [ims[i % len(ims)] for i in range(n)]


def median_time(fn, iters):
    fn()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def cfg(name):
    return json.loads((GOLDEN / "configs" / name).read_text())


def main():
    threads = [int(t) for t in sys.argv[1:]] or [1, 4]
    typical_names = ["photo_640x480.png", "coffee.png", "astronaut.png", "hifreq_333x517.png", "photo_500x375.jpg"]
    typical = load(typical_names, 64)
    large = load(["huge_3000x2000.png"], 8)
    files = [str(GOLDEN / "images" / typical_names[i % len(typical_names)]) for i in range(64)]
    rows = []
    for t in threads:
        torch.set_num_threads(t)
        for label, cfg_file, base in [("CLIP B/32", "openai_clip-vit-base-patch32.json", "CLIPImageProcessor"),
                                      ("SigLIP so400m-384", "google_siglip-so400m-patch14-384.json",
                                       "SiglipImageProcessor"),
                                      ("Qwen2-VL-2B", "Qwen_Qwen2-VL-2B-Instruct.json", "Qwen2VLImageProcessor")]:
            c = cfg(cfg_file)
            tv = getattr(transformers, base).from_dict(dict(c))
            pil = getattr(transformers, base + "Pil").from_dict(dict(c))
            ours = hpr.AutoProcessor.from_dict(c)
            ours_pil = hpr.AutoProcessor.from_dict(c, backend="pil")
            for blabel, batch, iters in [("64 x ~0.3MP", typical, 7), ("8 x 6MP", large, 3)]:
                if base.startswith("Qwen") and blabel == "8 x 6MP":
                    continue  # 6MP -> ~6MP output (36M floats per image); skip
                r = dict(threads=t, config=label, batch=blabel)
                r["transformers torchvision"] = median_time(lambda: tv(batch, return_tensors="np"), iters)
                r["transformers pil"] = median_time(lambda: pil(batch, return_tensors="np"), iters)
                r["rust torchvision"] = median_time(lambda: ours(batch, num_threads=t), iters)
                r["rust pil"] = median_time(lambda: ours_pil(batch, num_threads=t), iters)
                r["n"] = len(batch)
                rows.append(r)
                print(json.dumps(r), flush=True)
            if base == "CLIPImageProcessor":
                def py_files():
                    ims = [Image.open(f).convert("RGB") for f in files]
                    return tv(ims, return_tensors="np")
                r = dict(threads=t, config=label, batch="64 files (decode included)", n=64)
                r["transformers torchvision"] = median_time(py_files, 5)
                r["rust torchvision"] = median_time(lambda: ours(files, num_threads=t), 5)
                rows.append(r)
                print(json.dumps(r), flush=True)

    cols = ["transformers torchvision", "transformers pil", "rust torchvision", "rust pil"]
    print(f"\ntransformers {transformers.__version__}, torch {torch.__version__}, hf_processors_rs {hpr.__version__}")
    print("| threads | config | batch | " + " | ".join(cols) + " | speedup vs torchvision |")
    print("|---|---|---|" + "---|" * (len(cols) + 1))
    for r in rows:
        cells = [f"{r[c] * 1e3:.0f} ms" if c in r else "-" for c in cols]
        print(f"| {r['threads']} | {r['config']} | {r['batch']} | " + " | ".join(cells) +
              f" | {r['transformers torchvision'] / r['rust torchvision']:.1f}x |")


if __name__ == "__main__":
    main()
