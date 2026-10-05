"""Write seed corpora for the fuzz targets (needs Pillow and numpy).

    python fuzz/make_seeds.py [FUZZ_DIR]      # default: fuzz/

Seeds are small synthetic images in every PNG mode / bit depth Pillow can write (palettes with
transparency, 16-bit, interlaced) and JPEG variants (baseline, progressive, grayscale, CMYK,
chroma subsampling, restart markers, optimized Huffman tables), plus the golden configs.
"""

import io
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(__file__).resolve().parent
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root
rng = np.random.default_rng(0)


def rgb(w, h):
    y, x = np.mgrid[0:h, 0:w]
    a = np.stack([x * 255 // max(w - 1, 1), y * 255 // max(h - 1, 1), (x ^ y) & 255], -1)
    return (a + rng.integers(0, 32, a.shape)).clip(0, 255).astype(np.uint8)


def save(target, name, img, **kw):
    d = out / "corpus" / target
    d.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    img.save(buf, **kw)
    (d / name).write_bytes(buf.getvalue())


base = Image.fromarray(rgb(23, 17))
pngs = {
    "rgb": base,
    "l": base.convert("L"),
    "la": base.convert("LA"),
    "rgba": base.convert("RGBA"),
    "p": base.convert("P", palette=Image.Palette.ADAPTIVE, colors=16),
    "1": base.convert("1"),
    "i16": Image.fromarray(rgb(23, 17)[..., 0].astype(np.uint16) * 257),
    "tiny": Image.fromarray(rgb(1, 1)),
    "wide": Image.fromarray(rgb(300, 2)),
}
ptrans = base.convert("P", palette=Image.Palette.ADAPTIVE, colors=8)
for target in ("decode_pure", "decode_libjpeg"):
    for name, im in pngs.items():
        if target == "decode_pure":
            save(target, f"{name}.png", im, format="PNG")
    if target == "decode_pure":
        save(target, "p_trns.png", ptrans, format="PNG", transparency=0)
        save(target, "interlaced.png", base, format="PNG", interlace=1)
    jpegs = {
        "baseline_420.jpg": dict(quality=75, subsampling=2),
        "baseline_444.jpg": dict(quality=90, subsampling=0),
        "baseline_422.jpg": dict(quality=50, subsampling=1),
        "progressive.jpg": dict(quality=80, progressive=True),
        "optimized.jpg": dict(quality=60, optimize=True),
        "restart.jpg": dict(quality=70, restart_marker_blocks=1),
    }
    for name, kw in jpegs.items():
        save(target, name, base, format="JPEG", **kw)
    save(target, "gray.jpg", base.convert("L"), format="JPEG", quality=70)
    save(target, "cmyk.jpg", base.convert("CMYK"), format="JPEG", quality=70)
    save(target, "tiny.jpg", Image.fromarray(rgb(1, 1)), format="JPEG")
    save(target, "odd_9x33.jpg", Image.fromarray(rgb(9, 33)), format="JPEG", subsampling=2)

cfg = out / "corpus" / "config"
cfg.mkdir(parents=True, exist_ok=True)
for f in (root.parent / "golden" / "configs").glob("*.json"):
    shutil.copy(f, cfg / f.name)
print("seeds written to", out / "corpus")
