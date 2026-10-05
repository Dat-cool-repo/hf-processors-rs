"""Live parity against an installed `transformers` (skipped when it is not installed).

Same configs and images as the golden fixtures, compared value by value: the fixed-size
processors and Qwen2-VL must match exactly (max |diff| == 0), Whisper to float32 STFT noise.
"""

import json

import numpy as np
import pytest
from PIL import Image

import hf_processors_rs as hpr
from conftest import GOLDEN, load_manifest

transformers = pytest.importorskip("transformers")
pytest.importorskip("torchvision")

MANIFEST = load_manifest() if (GOLDEN / "manifest.json").exists() else {"cases": [], "images": []}
CONFIGS = sorted({(c["config"], c["processor"]) for c in MANIFEST["cases"]})
QWEN_CONFIGS = ["Qwen_Qwen2-VL-2B-Instruct.json", "Qwen_Qwen2.5-VL-7B-Instruct.json", "Qwen_Qwen3-VL-2B-Instruct.json"]
IMAGES = ["photo_640x480.png", "hifreq_333x517.png", "gray_300x200.png", "rgba_257x129.png", "la_100x150.png",
          "tiny_1x1.png", "tiny_7x5.png", "wide_1024x64.png", "tall_50x700.png", "astronaut.png", "coffee.png",
          "photo_500x375.jpg"]


def _open(name):
    im = Image.open(GOLDEN / "images" / name)
    im.load()
    return im


def _pair(cfg_file, base, backend):
    cfg = json.loads((GOLDEN / "configs" / cfg_file).read_text())
    ref = getattr(transformers, base + ("" if backend == "torchvision" else "Pil")).from_dict(dict(cfg))
    ours = hpr.AutoProcessor.from_dict(cfg, backend=backend, processor_type=base)
    return ref, ours


@pytest.mark.parametrize("backend", ["torchvision", "pil"])
@pytest.mark.parametrize("cfg_file,base", CONFIGS)
def test_fixed_processors_exact(golden, cfg_file, base, backend):
    ref, ours = _pair(cfg_file, base, backend)
    n = 0
    for name in IMAGES:
        im = _open(name)
        try:
            expected = ref(im, return_tensors="np")["pixel_values"]
        except Exception:  # transformers raises (e.g. ViT on grayscale): so must we
            with pytest.raises(ValueError):
                ours(im)
            continue
        got = ours(im)
        assert got.shape == expected.shape, name
        assert np.abs(got - expected).max() == 0.0, name
        n += 1
    # And as one batch (transformers stacks same-shape outputs).
    ok = [n for n in IMAGES if n not in ("gray_300x200.png", "rgba_257x129.png", "la_100x150.png")]
    if base not in ("ViTImageProcessor", "ConvNextImageProcessor"):
        ims = [_open(n) for n in ok]
        np.testing.assert_array_equal(ours(ims), ref(ims, return_tensors="np")["pixel_values"])
    assert n > 0


@pytest.mark.parametrize("backend", ["torchvision", "pil"])
@pytest.mark.parametrize("cfg_file", QWEN_CONFIGS)
def test_qwen2_vl_exact(golden, cfg_file, backend):
    ref, ours = _pair(cfg_file, "Qwen2VLImageProcessor", backend)
    ims = [_open(n) for n in IMAGES]
    expected = ref(ims, return_tensors="np")
    got = ours(ims)
    np.testing.assert_array_equal(got["image_grid_thw"], expected["image_grid_thw"])
    assert got["pixel_values"].shape == expected["pixel_values"].shape
    assert np.abs(got["pixel_values"] - expected["pixel_values"]).max() == 0.0


def test_whisper_close(golden):
    cfg = json.loads((GOLDEN / "configs" / "openai_whisper-tiny.json").read_text())
    ref = transformers.WhisperFeatureExtractor.from_dict(cfg)
    ours = hpr.AutoFeatureExtractor.from_dict(cfg)
    rng = np.random.default_rng(0)
    clips = [rng.normal(0, 0.1, n).astype(np.float32) for n in (16000, 52345)]
    for kw in [{}, dict(padding="longest", return_attention_mask=True), dict(do_normalize=True)]:
        e = ref(clips, sampling_rate=16000, return_tensors="np", **kw)
        g = ours(clips, sampling_rate=16000, **kw)
        assert np.abs(g["input_features"] - e["input_features"]).max() < 1e-4
        assert set(g) == set(e.keys())
        if "attention_mask" in e:
            np.testing.assert_array_equal(g["attention_mask"], e["attention_mask"])
