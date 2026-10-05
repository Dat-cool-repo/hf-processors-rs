"""Parity with transformers through the committed golden fixtures (no torch/transformers needed).

Fixed-size processors: every `ok` case of golden/manifest.json (8 Hub configs x 15 images x
2 backends) must reproduce transformers' uint8 tensor bit-for-bit (SHA-256), and the float
output must be exactly the backend's rescale+normalize LUT applied to it. The second suite
(golden/manifest_extra.json: resample filters, 16-bit PNG, CMYK JPEG, Qwen2-VL) is compared
through SHA-256 of the float32 pixel_values bytes.
"""

import json

import numpy as np
import pytest
from PIL import Image

import hf_processors_rs as hpr
from conftest import GOLDEN, invert_lut, load_manifest, sha256

MANIFEST = load_manifest() if (GOLDEN / "manifest.json").exists() else {"cases": [], "whisper": []}
EXTRA = load_manifest("manifest_extra.json") if (GOLDEN / "manifest_extra.json").exists() else {
    "fixed": [], "qwen": [], "whisper": []}

_procs = {}


def processor(config, backend, processor_type=None, overrides=None):
    key = (config, backend, processor_type, json.dumps(overrides, sort_keys=True))
    if key not in _procs:
        cfg = json.loads((GOLDEN / "configs" / config).read_text())
        cfg.update(overrides or {})
        _procs[key] = hpr.AutoProcessor.from_dict(cfg, backend=backend, processor_type=processor_type)
    return _procs[key]


def open_image(name):
    im = Image.open(GOLDEN / "images" / name)
    im.load()
    return im


_luts = {}


def lut_for(fixture, c):
    if fixture not in _luts:
        with np.load(GOLDEN / "fixtures" / fixture) as z:
            _luts[fixture] = {k: z[k] for k in z.files if k.startswith("lut_c")}
    return _luts[fixture][f"lut_c{c}"]


def case_id(c):
    return f"{c['case']}-{c['backend']}-{c['image']}"


OK_CASES = [c for c in MANIFEST["cases"] if c["status"] == "ok"]
ERR_CASES = [c for c in MANIFEST["cases"] if c["status"] == "error"]


@pytest.mark.parametrize("case", OK_CASES, ids=case_id)
def test_image_processor_bit_exact_pil_input(golden, case):
    proc = processor(case["config"], case["backend"], case["processor"])
    pv = proc(open_image(case["image"]))
    assert pv.shape == (1, *case["shape"])
    u8 = invert_lut(pv[0], lut_for(case["fixture"], case["shape"][0]))
    assert u8 is not None, "output values are not on the backend's LUT"
    assert sha256(u8) == case["sha256"]


PATH_CASES = [c for c in OK_CASES if c["image"] in ("photo_500x375.jpg", "astronaut.png", "palette_200x120.png",
                                                       "rgba_257x129.png", "gray_300x200.png")]


@pytest.mark.parametrize("case", PATH_CASES, ids=case_id)
def test_image_processor_file_path_input(golden, case):
    """Decoding in Rust (paths). JPEG is bit-exact when the wheel has the pil-jpeg feature."""
    proc = processor(case["config"], case["backend"], case["processor"])
    pv = proc(str(GOLDEN / "images" / case["image"]))
    u8 = invert_lut(pv[0], lut_for(case["fixture"], case["shape"][0]))
    if case["image"].endswith(".jpg") and not hpr.PIL_JPEG:
        pytest.skip("wheel built without pil-jpeg: JPEG decoding is within +-3 of Pillow")
    assert u8 is not None and sha256(u8) == case["sha256"]


@pytest.mark.parametrize("case", ERR_CASES, ids=case_id)
def test_errors_where_transformers_raises(golden, case):
    proc = processor(case["config"], case["backend"], case["processor"])
    if case["mode"] == "P":
        pytest.skip("documented divergence: palettes are expanded, transformers raises on palette indices")
    with pytest.raises(ValueError):
        proc(open_image(case["image"]))


EXTRA_FIXED = EXTRA["fixed"]


@pytest.mark.parametrize("case", EXTRA_FIXED, ids=lambda c: f"{c['suite']}-{c['backend']}-{c['image']}")
def test_extra_suite(golden, case):
    proc = processor(case["config"], case["backend"], case["processor"], case["overrides"])
    im = open_image(case["image"])
    if case["status"] == "error":
        if case["mode"] == "CMYK":
            pytest.skip("documented divergence: CMYK is converted to RGB, transformers raises without do_convert_rgb")
        with pytest.raises(ValueError):
            proc(im)
        return
    pv = proc(im)
    assert pv.shape == (1, *case["shape"])
    assert sha256(pv[0].astype(np.float32)) == case["sha256_f32"]


QWEN = EXTRA["qwen"]


@pytest.mark.parametrize("case", QWEN, ids=lambda c: f"{c['case']}-{c['backend']}-{c['image'][:40]}")
def test_qwen2_vl(golden, case):
    proc = processor(case["config"], case["backend"], None, case["overrides"])
    assert isinstance(proc, hpr.Qwen2VLImageProcessor)
    names = case.get("batch") or [case["image"]]
    ims = [open_image(n) for n in names]
    if case["status"] == "error":
        with pytest.raises(ValueError, match="aspect ratio"):
            proc(ims)
        return
    out = proc(ims)
    grid = case["grid_thw"] if "batch" in case else [case["grid_thw"]]
    assert out["image_grid_thw"].dtype == np.int64
    assert out["image_grid_thw"].tolist() == grid
    assert list(out["pixel_values"].shape) == case["shape"]
    assert sha256(out["pixel_values"]) == case["sha256_f32"]


def test_qwen2_vl_raw_float_fixture(golden):
    with np.load(GOLDEN / "fixtures" / "extra_qwen_raw.npz") as z:
        for key in z.files:
            case, backend, img = key.split("__")
            cfg = {"qwen2-vl-2b": "Qwen_Qwen2-VL-2B-Instruct.json", "qwen3-vl-2b": "Qwen_Qwen3-VL-2B-Instruct.json"}[case]
            out = processor(cfg, backend)(open_image(img.replace("_png", ".png")))
            np.testing.assert_array_equal(out["pixel_values"], z[key])


def _whisper_clips():
    with np.load(GOLDEN / "fixtures" / "whisper.npz") as z:
        a, b = z["pcm__short_1s"], z["pcm__speechlike_7p3s"]
    return {"a": a.astype(np.float32) / 32768.0, "b": b[:40000].astype(np.float32) / 32768.0,
            "short_1s": a.astype(np.float32) / 32768.0, "speechlike_7p3s": b.astype(np.float32) / 32768.0}


@pytest.mark.parametrize("case", EXTRA["whisper"], ids=lambda c: c["name"])
def test_whisper_options(golden, case):
    clips = _whisper_clips()
    fe = processor("openai_whisper-tiny.json", "torchvision", None, case["overrides"])
    waves = [clips[c] for c in case["clips"]]
    kw = dict(case["kwargs"])
    out = fe(waves if len(waves) > 1 else waves[0], sampling_rate=16000,
             dither_seed=case["dither_seed"] or 0, **kw)
    with np.load(GOLDEN / "fixtures" / "extra_whisper.npz") as z:
        ref = z[f"features__{case['name']}"]
    assert out["input_features"].shape == ref.shape
    # torch's float32 STFT vs our f64 STFT (see README): ~2e-5.
    assert np.abs(out["input_features"] - ref).max() <= 1e-4
    if "mask_shape" in case:
        m = out["attention_mask"]
        assert list(m.shape) == case["mask_shape"]
        assert m.sum(-1).tolist() == case["mask_lengths"]
    else:
        assert "attention_mask" not in out


def test_whisper_default(golden):
    clips = _whisper_clips()
    fe = hpr.AutoFeatureExtractor.from_pretrained(GOLDEN / "configs" / "openai_whisper-tiny.json")
    with np.load(GOLDEN / "fixtures" / "whisper.npz") as z:
        for name in ("short_1s", "speechlike_7p3s"):
            feats = fe(clips[name], sampling_rate=16000)["input_features"]
            assert feats.shape == (1, 80, 3000)
            assert np.abs(feats[0] - z[f"numpy__{name}"]).max() <= 1e-6
            assert np.abs(feats[0] - z[f"torch__{name}"]).max() <= 1e-4
