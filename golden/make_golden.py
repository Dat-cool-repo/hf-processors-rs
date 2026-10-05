#!/usr/bin/env python3
"""Generate golden fixtures for hf-processors-rs from Python `transformers`.

Run inside the project venv (see README):

    python golden/make_golden.py

Outputs (all under golden/):
  images/*.png, images/*.jpg   synthetic test inputs (astronaut.png and coffee.png are
                               public-domain / CC0 photos copied from scikit-image 0.19.3,
                               not generated; see golden/SOURCES.md)
  fixtures/<case>.npz          per (config, backend): reference outputs, losslessly compressed
  fixtures/whisper.npz         Whisper log-mel references
  manifest.json                case list, expected shapes / errors, library versions

Compression scheme (lossless, verified here): every image processor in scope ends with
`pixel_values = f(u8_image)` applied per channel, where `u8_image` is the resized/cropped
uint8 image and `f` is the backend's rescale+normalize. We store `u8` (CHW) and the LUT
`f(0..255)` computed with the backend's own rescale/normalize functions, and assert here
that `lut[c][u8] == pixel_values` bit-for-bit. The Rust tests rebuild the exact reference
float tensor from that.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
import transformers
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "images"
FIX_DIR = ROOT / "fixtures"
CFG_DIR = ROOT / "configs"

torch.set_num_threads(4)

# --------------------------------------------------------------------------------------
# Test images
# --------------------------------------------------------------------------------------


def _photo_like(rng, w, h, noise=10.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r = 128 + 100 * np.sin(xx / 37.0) * np.cos(yy / 53.0)
    g = xx * 255.0 / max(w - 1, 1)
    b = yy * 255.0 / max(h - 1, 1)
    arr = np.stack([r, g, b], -1)
    if noise:
        arr = arr + rng.normal(0, noise, arr.shape)
    img = Image.fromarray(np.clip(np.round(arr), 0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img)
    for _ in range(12):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        x1, y1 = x0 + int(rng.integers(1, max(2, w // 3))), y0 + int(rng.integers(1, max(2, h // 3)))
        col = tuple(int(c) for c in rng.integers(0, 256, 3))
        if rng.random() < 0.5:
            d.ellipse([x0, y0, x1, y1], fill=col)
        else:
            d.line([x0, y0, x1, y1], fill=col, width=int(rng.integers(1, 6)))
    d.text((w // 10, h // 10), "hf-processors-rs", fill=(255, 255, 255))
    return img


def make_images():
    IMG_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(20261004)
    out = {}

    out["photo_640x480.png"] = _photo_like(rng, 640, 480)

    # High-frequency content (1px checkerboard + noise) stresses bicubic overshoot/clamping.
    h, w = 517, 333
    yy, xx = np.mgrid[0:h, 0:w]
    checker = ((xx + yy) % 2) * 255
    arr = np.stack([checker, 255 - checker, (xx * 7 + yy * 3) % 256], -1).astype(np.float64)
    arr += rng.normal(0, 20, arr.shape)
    out["hifreq_333x517.png"] = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

    g = np.clip(np.mgrid[0:200, 0:300][1] * 0.85 + rng.normal(0, 15, (200, 300)), 0, 255).astype(np.uint8)
    out["gray_300x200.png"] = Image.fromarray(g, "L")

    rgb = np.asarray(_photo_like(rng, 257, 129))
    alpha = np.tile(np.linspace(0, 255, 257).astype(np.uint8), (129, 1))
    alpha[:20] = 0
    out["rgba_257x129.png"] = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")

    la = np.dstack([g[:150, :100], np.full((150, 100), 128, np.uint8)])
    out["la_100x150.png"] = Image.fromarray(la, "LA")

    pal = _photo_like(rng, 200, 120).quantize(colors=64, method=Image.Quantize.MEDIANCUT)
    out["palette_200x120.png"] = pal

    out["tiny_1x1.png"] = Image.fromarray(np.array([[[200, 30, 90]]], np.uint8), "RGB")
    out["tiny_2x3.png"] = Image.fromarray(rng.integers(0, 256, (3, 2, 3)).astype(np.uint8), "RGB")
    out["tiny_7x5.png"] = Image.fromarray(rng.integers(0, 256, (5, 7, 3)).astype(np.uint8), "RGB")
    out["wide_1024x64.png"] = _photo_like(rng, 1024, 64)
    out["tall_50x700.png"] = _photo_like(rng, 50, 700)

    # Huge input: no noise so the PNG stays small.
    out["huge_3000x2000.png"] = _photo_like(rng, 3000, 2000, noise=0)

    for name, img in out.items():
        img.save(IMG_DIR / name, optimize=True)

    # JPEG: exercises decoder differences (libjpeg-turbo vs the `image` crate), reported separately.
    _photo_like(rng, 500, 375).save(IMG_DIR / "photo_500x375.jpg", quality=90)

    names = sorted(out) + ["astronaut.png", "coffee.png", "photo_500x375.jpg"]
    return names


# --------------------------------------------------------------------------------------
# Image processors
# --------------------------------------------------------------------------------------

# (case id, config file, class base name, images subset or None for all)
SMALL = None
SUBSET = ["photo_640x480.png", "hifreq_333x517.png", "gray_300x200.png", "tiny_2x3.png", "astronaut.png",
          "tall_50x700.png"]
CONFIGS = [
    ("clip-vit-base-patch32", "openai_clip-vit-base-patch32.json", "CLIPImageProcessor", SMALL),
    ("clip-vit-large-patch14-336", "openai_clip-vit-large-patch14-336.json", "CLIPImageProcessor", SUBSET),
    ("vit-base-patch16-224", "google_vit-base-patch16-224.json", "ViTImageProcessor", SMALL),
    ("siglip-base-patch16-224", "google_siglip-base-patch16-224.json", "SiglipImageProcessor", SMALL),
    ("siglip-so400m-patch14-384", "google_siglip-so400m-patch14-384.json", "SiglipImageProcessor", SUBSET),
    ("convnext-tiny-224", "facebook_convnext-tiny-224.json", "ConvNextImageProcessor", SMALL),
    ("convnext-base-384", "facebook_convnext-base-384.json", "ConvNextImageProcessor", SUBSET),
    ("blip-image-captioning-base", "Salesforce_blip-image-captioning-base.json", "BlipImageProcessor", SUBSET),
]
BACKENDS = ["torchvision", "pil"]


def processor_class(base: str, backend: str):
    name = base if backend == "torchvision" else base + "Pil"
    return getattr(transformers, name)


def build_lut(proc, backend: str, channels: int) -> np.ndarray:
    """f(0..255) per channel using the backend's own rescale/normalize code."""
    ramp = np.tile(np.arange(256, dtype=np.uint8), (channels, 1, 1))  # (C, 1, 256)
    kw = dict(do_rescale=proc.do_rescale, rescale_factor=proc.rescale_factor, do_normalize=proc.do_normalize,
              image_mean=proc.image_mean, image_std=proc.image_std)
    if backend == "torchvision":
        t = torch.from_numpy(ramp)[None]  # (1, C, 1, 256)
        out = proc.rescale_and_normalize(t, kw["do_rescale"], kw["rescale_factor"], kw["do_normalize"],
                                         kw["image_mean"], kw["image_std"])
        out = out[0].numpy() if out.dtype == torch.float32 else out[0].float().numpy()
    else:
        img = ramp
        if kw["do_rescale"]:
            img = proc.rescale(img, kw["rescale_factor"])
        if kw["do_normalize"]:
            img = proc.normalize(img, kw["image_mean"], kw["image_std"])
        out = np.asarray(img, dtype=np.float32)
    return out[:, 0, :].astype(np.float32)  # (C, 256)


def invert(pv: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Recover the uint8 image from pixel_values; assert lossless."""
    c = pv.shape[0]
    u8 = np.empty(pv.shape, np.uint8)
    for ch in range(c):
        table = {lut[ch, v].tobytes(): v for v in range(256)}
        assert len(table) == 256, "LUT is not injective"
        flat = pv[ch].reshape(-1)
        u8[ch].reshape(-1)[:] = [table[x.tobytes()] for x in flat]
    rebuilt = lut[np.arange(c)[:, None], u8.reshape(c, -1).astype(np.int64)].reshape(pv.shape)
    assert np.array_equal(rebuilt.view(np.uint32), pv.view(np.uint32)), "lossy"
    return u8


# Full uint8 references kept in the repo (everything else is verified by SHA-256 in the repo,
# with full arrays optionally written to EXTERNAL_DIR for diagnosing mismatches).
# The JPEG case is always kept in full: the pure-Rust JPEG decoder is not bit-exact, so the
# Rust tests need the full reference to check its bounded difference.
FULL_IN_REPO = {
    "clip-vit-base-patch32": ["photo_640x480.png", "hifreq_333x517.png", "astronaut.png", "tiny_2x3.png",
                              "photo_500x375.jpg"],
    "vit-base-patch16-224": ["photo_640x480.png", "hifreq_333x517.png", "astronaut.png", "tiny_2x3.png",
                             "photo_500x375.jpg"],
    "siglip-base-patch16-224": ["photo_500x375.jpg"],
    "convnext-tiny-224": ["photo_500x375.jpg"],
}
# Optional: set HF_PROCESSORS_GOLDEN_FULL to a directory outside the repo to also keep the full
# uint8 arrays (the Rust tests use them for per-pixel diagnostics when a case fails).
EXTERNAL_DIR = Path(os.environ["HF_PROCESSORS_GOLDEN_FULL"]) if os.environ.get("HF_PROCESSORS_GOLDEN_FULL") else None


def save_external(name: str, **arrays):
    """Write full reference arrays to EXTERNAL_DIR (no-op when HF_PROCESSORS_GOLDEN_FULL is unset)."""
    if EXTERNAL_DIR is None:
        return
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EXTERNAL_DIR / name, **arrays)


def run_image_cases(image_names):
    FIX_DIR.mkdir(exist_ok=True)
    cases = []
    for case_id, cfg_file, base, subset in CONFIGS:
        cfg = json.loads((CFG_DIR / cfg_file).read_text())
        names = image_names if subset is None else subset
        for backend in BACKENDS:
            cls = processor_class(base, backend)
            proc = cls.from_dict(dict(cfg))
            repo_arrays, ext_arrays, luts = {}, {}, {}
            fixture = f"{case_id}__{backend}.npz"
            for name in names:
                with Image.open(IMG_DIR / name) as im:
                    im.load()
                    mode = im.mode
                    try:
                        pv = proc(im, return_tensors="np")["pixel_values"][0]
                    except Exception as e:  # noqa: BLE001 - we record transformers' failures
                        cases.append(dict(case=case_id, config=cfg_file, processor=base, backend=backend,
                                          image=name, mode=mode, status="error", error=f"{type(e).__name__}: {e}"))
                        continue
                pv = np.ascontiguousarray(pv, dtype=np.float32)
                c = pv.shape[0]
                if c not in luts:
                    luts[c] = build_lut(proc, backend, c)
                u8 = np.ascontiguousarray(invert(pv, luts[c]))
                key = name.replace(".", "_")
                in_repo = name in FULL_IN_REPO.get(case_id, [])
                (repo_arrays if in_repo else ext_arrays)[f"u8__{key}"] = u8
                cases.append(dict(case=case_id, config=cfg_file, processor=base, backend=backend, image=name,
                                  mode=mode, status="ok", shape=list(pv.shape), key=key, fixture=fixture,
                                  sha256=hashlib.sha256(u8.tobytes()).hexdigest(),
                                  full="repo" if in_repo else "external"))
            for c, lut in luts.items():
                repo_arrays[f"lut_c{c}"] = lut
            np.savez_compressed(FIX_DIR / fixture, **repo_arrays)
            save_external(fixture, **ext_arrays, **repo_arrays)
            print(f"{case_id:28s} {backend:12s} {len(names)} images", flush=True)
    return cases


def sanity_raw_fixture():
    """One uncompressed float reference (CLIP, torchvision, astronaut) as an independent check."""
    cfg = json.loads((CFG_DIR / "openai_clip-vit-base-patch32.json").read_text())
    proc = transformers.CLIPImageProcessor.from_dict(cfg)
    with Image.open(IMG_DIR / "astronaut.png") as im:
        pv = proc(im, return_tensors="np")["pixel_values"][0].astype(np.float32)
    np.savez_compressed(FIX_DIR / "raw_clip_torchvision_astronaut.npz", pixel_values=pv)


# --------------------------------------------------------------------------------------
# Whisper
# --------------------------------------------------------------------------------------


def lcg_noise(n, seed=12345):
    """Deterministic uniform noise in [-1, 1) reproducible in Rust (64-bit LCG)."""
    out = np.empty(n, np.float64)
    s = seed
    for i in range(n):
        s = (s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        out[i] = ((s >> 11) / float(1 << 53)) * 2.0 - 1.0
    return out


def make_clip(seconds: float, sr=16000):
    n = int(round(seconds * sr))
    t = np.arange(n) / sr
    x = 0.3 * np.sin(2 * np.pi * 440.0 * t) + 0.2 * np.sin(2 * np.pi * (150.0 + 400.0 * t) * t)
    x += 0.05 * lcg_noise(n)
    x *= np.minimum(1.0, t * 4.0)  # fade-in
    return (np.round(x * 32767.0)).astype(np.int16)


def run_whisper():
    from transformers import WhisperFeatureExtractor

    cfg = json.loads((CFG_DIR / "openai_whisper-tiny.json").read_text())
    fe = WhisperFeatureExtractor.from_dict(cfg)
    base = make_clip(7.3)
    clips = {
        "short_1s": make_clip(1.0),
        "speechlike_7p3s": base,
        "long_31s": np.tile(base, 5)[: 31 * 16000],  # exercises truncation to 30 s
    }
    arrays, cases = {}, []
    for name, pcm in clips.items():
        wave = pcm.astype(np.float32) / 32768.0
        feats = fe(wave, sampling_rate=16000, return_tensors="np")["input_features"][0]  # torch path
        # numpy (float64 FFT) path for comparison:
        padded = np.zeros(fe.n_samples, np.float32)
        padded[: min(len(wave), fe.n_samples)] = wave[: fe.n_samples]
        feats_np = fe._np_extract_fbank_features(padded[None], "cpu")[0]
        if name != "long_31s":
            arrays[f"pcm__{name}"] = pcm
        arrays[f"torch__{name}"] = feats.astype(np.float32)
        arrays[f"numpy__{name}"] = feats_np.astype(np.float32)
        cases.append(dict(name=name, samples=int(len(pcm)), shape=list(feats.shape)))
    arrays["mel_filters"] = fe.mel_filters.astype(np.float64)
    np.savez_compressed(FIX_DIR / "whisper.npz", **arrays)
    print("whisper", [c["name"] for c in cases])
    return cases


def composite_cases():
    """transformers 4.4x-4.5x `convert_to_rgb`: alpha-composite on white (Pillow integer math)."""
    out = []
    for name in ["rgba_257x129.png", "la_100x150.png", "palette_200x120.png", "gray_300x200.png"]:
        with Image.open(IMG_DIR / name) as im:
            rgba = im.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255))
            rgb = np.ascontiguousarray(np.asarray(Image.alpha_composite(white, rgba).convert("RGB")))
        out.append(dict(image=name, sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
    return out


def main():
    names = make_images()
    cases = run_image_cases(names)
    sanity_raw_fixture()
    whisper_cases = run_whisper()
    manifest = dict(
        versions=dict(transformers=transformers.__version__, pillow=PIL.__version__, torch=torch.__version__,
                      torchvision=torchvision.__version__, numpy=np.__version__, python=sys.version.split()[0],
                      torch_cpu_capability=torch.backends.cpu.get_cpu_capability()),
        images=names,
        cases=cases,
        whisper=whisper_cases,
        composite=composite_cases(),
    )
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    n_ok = sum(c["status"] == "ok" for c in cases)
    print(f"{n_ok} ok cases, {len(cases) - n_ok} expected-error cases")


if __name__ == "__main__":
    main()
