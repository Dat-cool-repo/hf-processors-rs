#!/usr/bin/env python3
"""Second golden suite for hf-processors-rs (run after make_golden.py, same venv):

    python golden/make_golden_extra.py

Covers what the first suite does not:
  * resample filters: LANCZOS / NEAREST on torchvision; NEAREST / BOX / HAMMING / LANCZOS on PIL
    (BOX / HAMMING on torchvision raise in transformers, which is recorded as an expected error);
  * 16-bit PNGs (I;16 gray, gray+alpha, RGB, RGBA) and a CMYK JPEG, decoded like Pillow;
  * palette PNGs without do_convert_rgb (Rust expands the palette: compared against transformers
    on `img.convert("RGB")`);
  * Qwen2-VL / Qwen2.5-VL / Qwen3-VL (`Qwen2VLImageProcessor[Pil]`): smart_resize, patching,
    image_grid_thw, batching, the aspect-ratio error;
  * WhisperFeatureExtractor options: padding / truncation / max_length / pad_to_multiple_of,
    do_normalize, return_attention_mask (incl. its sample-level quirk) and seeded dither.

Image cases are checked bit-exactly through SHA-256 of the float32 `pixel_values` bytes; the
uint8 tensors behind them (rebuilt losslessly with the rescale/normalize LUT, as in
make_golden.py) go to $HF_PROCESSORS_GOLDEN_FULL (when set) for per-pixel diagnostics, LUTs to the repo.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import make_golden as mg  # noqa: E402

IMG_DIR, FIX_DIR, CFG_DIR, EXTERNAL_DIR = mg.IMG_DIR, mg.FIX_DIR, mg.CFG_DIR, mg.EXTERNAL_DIR


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# --------------------------------------------------------------------------------------
# Extra test images
# --------------------------------------------------------------------------------------


def write_png16(path: Path, arr: np.ndarray, color_type: int):
    """Minimal 16-bit PNG writer (Pillow cannot write 16-bit RGB/RGBA/LA)."""
    h, w = arr.shape[:2]
    raw = b"".join(b"\x00" + arr[y].astype(">u2").tobytes() for y in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 16, color_type, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) +
                     chunk(b"IEND", b""))


def make_extra_images():
    rng = np.random.default_rng(424242)
    out = []
    # I;16 gray: mostly <= 255 (kept) with some larger values (clipped to 255 by Pillow).
    g = np.mgrid[0:120, 0:160][1] * 2.0 + rng.normal(0, 20, (120, 160))
    g[::7] *= 3
    write_png16(IMG_DIR / "gray16_160x120.png", np.clip(g, 0, 65535).astype(np.uint16), 0)
    base = np.asarray(mg._photo_like(rng, 150, 100)).astype(np.uint16)
    rgb16 = base * 257 + rng.integers(0, 257, base.shape).astype(np.uint16)  # full 16-bit range
    write_png16(IMG_DIR / "rgb16_150x100.png", rgb16, 2)
    a = np.tile(np.linspace(0, 65535, 150).astype(np.uint16), (100, 1))[..., None]
    write_png16(IMG_DIR / "rgba16_150x100.png", np.concatenate([rgb16, a], -1), 6)
    la = np.stack([rgb16[..., 1], np.full((100, 150), 40000, np.uint16)], -1)
    write_png16(IMG_DIR / "la16_150x100.png", la, 4)
    cmyk = np.asarray(mg._photo_like(rng, 200, 150).convert("CMYK")).copy()
    cmyk[..., 3] = np.clip(np.mgrid[0:150, 0:200][0] * 1.5, 0, 255).astype(np.uint8)  # some black
    Image.fromarray(cmyk, "CMYK").save(IMG_DIR / "cmyk_200x150.jpg", quality=90)
    # Aspect ratio 201 -> smart_resize raises.
    Image.fromarray(rng.integers(0, 256, (2, 402, 3)).astype(np.uint8), "RGB").save(IMG_DIR / "extreme_402x2.png")
    out = ["gray16_160x120.png", "rgb16_150x100.png", "rgba16_150x100.png", "la16_150x100.png",
           "cmyk_200x150.jpg", "extreme_402x2.png"]
    return out


# --------------------------------------------------------------------------------------
# Generic runner for fixed-size image processors
# --------------------------------------------------------------------------------------

SUITES = []  # (suite, config file, class base, overrides, images, input transform)
RESAMPLE_IMAGES = ["photo_640x480.png", "hifreq_333x517.png", "tiny_7x5.png", "wide_1024x64.png", "astronaut.png",
                   "gray_300x200.png"]
for r, name in [(0, "nearest"), (1, "lanczos"), (4, "box"), (5, "hamming")]:
    SUITES.append((f"resample-{name}", "openai_clip-vit-base-patch32.json", "CLIPImageProcessor", {"resample": r},
                   RESAMPLE_IMAGES, None))
# Bilinear on the "large" CLIP config (crop 336) to also cover bilinear up/down on PIL+torch.
SUITES.append(("resample-bilinear", "openai_clip-vit-large-patch14-336.json", "CLIPImageProcessor", {"resample": 2},
               RESAMPLE_IMAGES, None))
DECODE_IMAGES = ["gray16_160x120.png", "rgb16_150x100.png", "rgba16_150x100.png", "la16_150x100.png",
                 "cmyk_200x150.jpg"]
for case, cfg, base in [("decode-clip", "openai_clip-vit-base-patch32.json", "CLIPImageProcessor"),
                        ("decode-siglip", "google_siglip-base-patch16-224.json", "SiglipImageProcessor"),
                        ("decode-vit", "google_vit-base-patch16-224.json", "ViTImageProcessor")]:
    SUITES.append((case, cfg, base, {}, DECODE_IMAGES, None))
# Palette PNG without do_convert_rgb: Rust expands the palette, which equals transformers on
# image.convert("RGB") (transformers itself raises on the palette indices).
for case, cfg, base in [("palette-vit", "google_vit-base-patch16-224.json", "ViTImageProcessor"),
                        ("palette-convnext", "facebook_convnext-tiny-224.json", "ConvNextImageProcessor")]:
    SUITES.append((case, cfg, base, {}, ["palette_200x120.png"], "convert_rgb"))


def load_input(name: str, transform: str | None):
    im = Image.open(IMG_DIR / name)
    im.load()
    if transform == "convert_rgb":
        im = im.convert("RGB")
    return im


def run_fixed_suites():
    cases = []
    for suite, cfg_file, base, overrides, images, transform in SUITES:
        cfg = json.loads((CFG_DIR / cfg_file).read_text())
        cfg.update(overrides)
        for backend in mg.BACKENDS:
            proc = mg.processor_class(base, backend).from_dict(dict(cfg))
            luts, ext = {}, {}
            fixture = f"extra_{suite}__{backend}.npz"
            for name in images:
                im = load_input(name, transform)
                common = dict(suite=suite, config=cfg_file, overrides=overrides, processor=base, backend=backend,
                              image=name, mode=Image.open(IMG_DIR / name).mode, input_transform=transform)
                try:
                    pv = proc(im, return_tensors="np")["pixel_values"][0]
                except Exception as e:  # noqa: BLE001
                    cases.append(dict(common, status="error", error=f"{type(e).__name__}: {e}"[:300]))
                    continue
                pv = np.ascontiguousarray(pv, dtype=np.float32)
                c = pv.shape[0]
                if c not in luts:
                    luts[c] = mg.build_lut(proc, backend, c)
                u8 = mg.invert(pv, luts[c])
                key = name.replace(".", "_")
                ext[f"u8__{key}"] = u8
                cases.append(dict(common, status="ok", shape=list(pv.shape), sha256_f32=sha(pv), sha256_u8=sha(u8),
                                  key=key, fixture=fixture))
            np.savez_compressed(FIX_DIR / fixture, **{f"lut_c{c}": v for c, v in luts.items()})
            mg.save_external(fixture, **ext, **{f"lut_c{c}": v for c, v in luts.items()})
            print(f"{suite:24s} {backend:12s} {len(images)} images", flush=True)
    return cases


def decode_reference():
    """What `Image.open(p).convert("RGB")` gives for the extra decode images (SHA of HWC bytes)."""
    out = []
    for name in DECODE_IMAGES + ["palette_200x120.png"]:
        with Image.open(IMG_DIR / name) as im:
            mode = im.mode
            rgb = np.ascontiguousarray(np.asarray(im.convert("RGB")))
        out.append(dict(image=name, mode=mode, shape=list(rgb.shape), sha256=sha(rgb)))
    mg.save_external("extra_decode.npz",
                     **{n.replace(".", "_"): np.asarray(Image.open(IMG_DIR / n).convert("RGB"))
                        for n in DECODE_IMAGES})
    return out


# --------------------------------------------------------------------------------------
# Qwen2-VL family
# --------------------------------------------------------------------------------------

QWEN_CONFIGS = [
    ("qwen2-vl-2b", "Qwen_Qwen2-VL-2B-Instruct.json", {}),
    ("qwen2.5-vl-7b", "Qwen_Qwen2.5-VL-7B-Instruct.json", {}),
    ("qwen3-vl-2b", "Qwen_Qwen3-VL-2B-Instruct.json", {}),
    # The commonly recommended budget (256-1280 visual tokens): exercises both the min_pixels
    # upscale and the max_pixels downscale branches of smart_resize.
    ("qwen2-vl-2b-budget", "Qwen_Qwen2-VL-2B-Instruct.json", {"min_pixels": 256 * 28 * 28, "max_pixels": 1280 * 28 * 28}),
]
QWEN_IMAGES = ["photo_640x480.png", "hifreq_333x517.png", "gray_300x200.png", "rgba_257x129.png", "tiny_2x3.png",
               "tiny_7x5.png", "wide_1024x64.png", "tall_50x700.png", "astronaut.png", "photo_500x375.jpg",
               "extreme_402x2.png"]
QWEN_HUGE = "huge_3000x2000.png"
QWEN_BATCH = ["photo_640x480.png", "gray_300x200.png", "tiny_2x3.png", "photo_640x480.png", "wide_1024x64.png"]
QWEN_RAW = [("qwen2-vl-2b", "torchvision", "tiny_7x5.png"), ("qwen3-vl-2b", "pil", "tiny_2x3.png")]


def qwen_class(backend):
    return transformers.Qwen2VLImageProcessor if backend == "torchvision" else transformers.Qwen2VLImageProcessorPil


def unpatchify(pv, grid, c, t, p, m):
    """Inverse of Qwen2VL patchify; asserts the temporal copies are identical."""
    _, gh, gw = (int(v) for v in grid)
    x = pv.reshape(gh // m, gw // m, m, m, c, t, p, p)
    assert np.array_equal(x.view(np.uint32), np.broadcast_to(x[:, :, :, :, :, :1], x.shape).view(np.uint32))
    x = x[:, :, :, :, :, 0]
    return np.ascontiguousarray(x.transpose(4, 0, 2, 5, 1, 3, 6).reshape(c, gh * p, gw * p))


def run_qwen():
    cases, raw = [], {}
    for case_id, cfg_file, overrides in QWEN_CONFIGS:
        cfg = json.loads((CFG_DIR / cfg_file).read_text())
        cfg.update(overrides)
        for backend in mg.BACKENDS:
            proc = qwen_class(backend).from_dict(dict(cfg))
            p, m, t = proc.patch_size, proc.merge_size, proc.temporal_patch_size
            luts, ext = {}, {}
            fixture = f"extra_{case_id}__{backend}.npz"
            images = QWEN_IMAGES + ([QWEN_HUGE] if case_id in ("qwen2-vl-2b", "qwen2-vl-2b-budget") else [])
            for name in images:
                im = load_input(name, None)
                common = dict(suite="qwen", case=case_id, config=cfg_file, overrides=overrides, backend=backend,
                              image=name, mode=im.mode)
                try:
                    out = proc(im, return_tensors="np")
                except Exception as e:  # noqa: BLE001
                    cases.append(dict(common, status="error", error=f"{type(e).__name__}: {e}"[:300]))
                    continue
                pv = np.ascontiguousarray(out["pixel_values"], dtype=np.float32)
                grid = out["image_grid_thw"][0].tolist()
                chw = unpatchify(pv, grid, 3, t, p, m)
                if 3 not in luts:
                    luts[3] = mg.build_lut(proc, backend, 3)
                u8 = mg.invert(chw, luts[3])
                key = name.replace(".", "_")
                ext[f"u8__{key}"] = u8
                cases.append(dict(common, status="ok", shape=list(pv.shape), grid_thw=grid, resized=list(u8.shape[1:]),
                                  sha256_f32=sha(pv), sha256_u8=sha(u8), key=key, fixture=fixture))
                if (case_id, backend, name) in QWEN_RAW:
                    raw[f"{case_id}__{backend}__{key}"] = pv
            # Batch call: concatenation order and image_grid_thw.
            ims = [load_input(n, None) for n in QWEN_BATCH]
            out = proc(ims, return_tensors="np")
            pv = np.ascontiguousarray(out["pixel_values"], dtype=np.float32)
            cases.append(dict(suite="qwen", case=case_id, config=cfg_file, overrides=overrides, backend=backend,
                              image="+".join(QWEN_BATCH), batch=QWEN_BATCH, status="ok", shape=list(pv.shape),
                              grid_thw=out["image_grid_thw"].tolist(), sha256_f32=sha(pv)))
            np.savez_compressed(FIX_DIR / fixture, **{f"lut_c{c}": v for c, v in luts.items()})
            mg.save_external(fixture, **ext, **{f"lut_c{c}": v for c, v in luts.items()})
            print(f"{case_id:24s} {backend:12s} {len(images)} images + batch", flush=True)
    np.savez_compressed(FIX_DIR / "extra_qwen_raw.npz", **raw)
    return cases


# --------------------------------------------------------------------------------------
# Whisper options
# --------------------------------------------------------------------------------------


def gaussian_noise_np(seed: int, n: int) -> np.ndarray:
    """Same stream as hf_processors::audio::whisper::gaussian_noise (SplitMix64 + Box-Muller)."""
    golden = np.uint64(0x9E3779B97F4A7C15)
    npairs = (n + 1) // 2
    i = np.arange(1, 2 * npairs + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(seed) + i * golden
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    u = (z >> np.uint64(11)).astype(np.float64)
    u1 = (u[0::2] + 1.0) * 2.0**-53
    u2 = u[1::2] * 2.0**-53
    r = np.sqrt(-2.0 * np.log(u1))
    th = 2.0 * np.pi * u2
    out = np.empty(2 * npairs)
    out[0::2] = r * np.cos(th)
    out[1::2] = r * np.sin(th)
    return out[:n].astype(np.float32)


# (name, clips, call kwargs, config overrides, dither seed). Clips: "a" = 1 s clip, "b" = the
# first 2.5 s of the 7.3 s clip (both rebuilt in Rust from whisper.npz).
WHISPER_CASES = [
    ("longest_mask", ["a", "b"], dict(padding="longest", return_attention_mask=True), {}, None),
    ("normalize_maxlen", ["a"], dict(do_normalize=True, max_length=32000), {}, None),
    ("normalize_longest_mask", ["a", "b"], dict(do_normalize=True, padding="longest", return_attention_mask=True),
     {}, None),
    ("no_truncation", ["b"], dict(max_length=32000, truncation=False), {}, None),
    ("truncation", ["b"], dict(max_length=32000), {}, None),
    ("pad_multiple", ["a"], dict(padding="longest", pad_to_multiple_of=3000, return_attention_mask=True), {}, None),
    ("do_not_pad", ["a"], dict(padding="do_not_pad"), {}, None),
    ("config_mask_default", ["a"], dict(padding="longest"), {"return_attention_mask": True}, None),
    ("dither", ["a", "b"], dict(padding="longest", return_attention_mask=True), {"dither": 0.01}, 0),
    ("dither_seed7_maxlen", ["a"], dict(max_length=24000), {"dither": 0.001}, 7),
]


def run_whisper_options():
    cfg = json.loads((CFG_DIR / "openai_whisper-tiny.json").read_text())
    with np.load(FIX_DIR / "whisper.npz") as z:
        clips = {"a": z["pcm__short_1s"], "b": z["pcm__speechlike_7p3s"][:40000]}
    arrays, cases = {}, []
    orig_randn = torch.randn
    for name, clip_ids, kwargs, overrides, seed in WHISPER_CASES:
        fe = transformers.WhisperFeatureExtractor.from_dict(dict(cfg, **overrides))
        waves = [clips[c].astype(np.float32) / 32768.0 for c in clip_ids]
        if seed is not None:
            def fake_randn(shape, dtype=None, device=None, _seed=seed, **_):
                n = int(np.prod(shape))
                return torch.from_numpy(gaussian_noise_np(_seed, n).reshape(tuple(shape))).to(dtype)
            torch.randn = fake_randn
        try:
            out = fe(waves if len(waves) > 1 else waves[0], sampling_rate=16000, return_tensors="np", **kwargs)
        finally:
            torch.randn = orig_randn
        feats = out["input_features"].astype(np.float32)
        arrays[f"features__{name}"] = feats
        case = dict(name=name, clips=clip_ids, kwargs=kwargs, overrides=overrides, dither_seed=seed,
                    shape=list(feats.shape))
        if "attention_mask" in out:
            mask = np.asarray(out["attention_mask"], np.int32)
            # Every mask is a prefix of ones per row; store the shape and the row sums.
            for row in mask:
                k = int(row.sum())
                assert row[:k].all() and not row[k:].any()
            case["mask_shape"] = list(mask.shape)
            case["mask_lengths"] = mask.sum(-1).tolist()
        cases.append(case)
        print("whisper", name, feats.shape, case.get("mask_shape"), flush=True)
    np.savez_compressed(FIX_DIR / "extra_whisper.npz", **arrays)
    return cases


def main():
    images = make_extra_images()
    fixed = run_fixed_suites()
    decode = decode_reference()
    qwen = run_qwen()
    whisper = run_whisper_options()
    manifest = dict(
        versions=dict(transformers=transformers.__version__, pillow=Image.__version__, torch=torch.__version__,
                      numpy=np.__version__, torch_cpu_capability=torch.backends.cpu.get_cpu_capability()),
        extra_images=images,
        fixed=fixed,
        decode=decode,
        qwen=qwen,
        whisper=whisper,
    )
    (ROOT / "manifest_extra.json").write_text(json.dumps(manifest, indent=1))
    for label, cs in [("fixed", fixed), ("qwen", qwen)]:
        n_ok = sum(c["status"] == "ok" for c in cs)
        print(f"{label}: {n_ok} ok cases, {len(cs) - n_ok} expected-error cases")
    print(f"whisper: {len(whisper)} cases")


if __name__ == "__main__":
    main()
