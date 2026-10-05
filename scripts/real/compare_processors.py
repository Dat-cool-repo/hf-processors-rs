"""Compare hf_processors_rs with transformers on real photos, both backends: CLIP
(pixel_values) and a Qwen2-VL family processor (pixel_values and image_grid_thw).

    python scripts/real/compare_processors.py IMAGE_DIR [repo ...]

Default repos: openai/clip-vit-base-patch32 Qwen/Qwen2-VL-2B-Instruct (configs only, from
the Hub). Both libraries get the file: transformers as `Image.open(path)`, hf_processors_rs
as the path (decoded in Rust with libjpeg-turbo, `PIL_JPEG`). Prints a JSON summary.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import transformers
from PIL import Image
from transformers import AutoImageProcessor

import hf_processors_rs as hpr

PIL_CLASSES = {"CLIPImageProcessor": "CLIPImageProcessorPil", "Qwen2VLImageProcessor": "Qwen2VLImageProcessorPil"}


def reference(repo, backend):
    proc = AutoImageProcessor.from_pretrained(repo)
    if backend == "pil":
        cls = getattr(transformers, PIL_CLASSES[type(proc).__name__])
        proc = cls.from_pretrained(repo)
    return proc


def main():
    image_dir = Path(sys.argv[1])
    repos = sys.argv[2:] or ["openai/clip-vit-base-patch32", "Qwen/Qwen2-VL-2B-Instruct"]
    files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    assert hpr.PIL_JPEG, "build hf_processors_rs with the default pil-jpeg feature"
    results = []
    for repo in repos:
        for backend in ("torchvision", "pil"):
            ref = reference(repo, backend)
            ours = hpr.AutoImageProcessor.from_pretrained(repo, backend=backend)
            t = time.time()
            n_same, max_diff, grids_same, patches = 0, 0.0, 0, 0
            bad = []
            for f in files:
                with Image.open(f) as im:
                    want = ref(im, return_tensors="np")
                got = ours(str(f))
                if isinstance(got, dict):
                    pv_w, pv_g = want["pixel_values"], got["pixel_values"]
                    g_ok = np.array_equal(np.asarray(want["image_grid_thw"]), got["image_grid_thw"])
                    grids_same += g_ok
                    patches += pv_g.shape[0]
                else:
                    pv_w, pv_g = want["pixel_values"], got
                    g_ok = True
                same = pv_w.shape == pv_g.shape and pv_w.tobytes() == pv_g.astype(np.float32).tobytes()
                if pv_w.shape == pv_g.shape:
                    max_diff = max(max_diff, float(np.abs(pv_w - pv_g).max()))
                n_same += same and g_ok
                if not (same and g_ok):
                    bad.append(f.name)
            r = dict(repo=repo, backend=backend, reference=type(ref).__name__, images=len(files),
                     identical=int(n_same), max_abs_diff=max_diff, seconds=round(time.time() - t, 1))
            if patches:
                r.update(grid_thw_identical=int(grids_same), total_patches=patches)
            if bad:
                r["mismatches"] = bad[:20]
            print(json.dumps(r), flush=True)
            results.append(r)
    print(json.dumps(dict(transformers=transformers.__version__, pillow=Image.__version__,
                          hf_processors_rs=hpr.__version__, results=results), indent=1))


if __name__ == "__main__":
    main()
