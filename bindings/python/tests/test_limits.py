"""Decompression-bomb limit (Pillow's MAX_IMAGE_PIXELS semantics) and robustness to hostile
configs and inputs."""

import io
import struct
import warnings
import zlib

import numpy as np
import pytest
from PIL import Image

import hf_processors_rs as hpr
from conftest import GOLDEN


def png_header_only(width, height):
    """A PNG whose IHDR declares width x height but which has a single empty IDAT chunk."""

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"")) + chunk(b"IEND", b"")


def jpeg_bytes(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 120, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def png_bytes(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 120, 200)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def clip(golden):
    return hpr.AutoImageProcessor.from_pretrained(golden / "configs" / "openai_clip-vit-base-patch32.json")


@pytest.fixture
def limit():
    """Restore the process-wide limit after the test."""
    old = hpr.get_max_image_pixels()
    yield hpr.set_max_image_pixels
    hpr.set_max_image_pixels(old)


def test_default_is_pillows():
    assert hpr.DEFAULT_MAX_IMAGE_PIXELS == Image.MAX_IMAGE_PIXELS == 89_478_485
    assert hpr.get_max_image_pixels() == hpr.DEFAULT_MAX_IMAGE_PIXELS
    assert issubclass(hpr.DecompressionBombError, ValueError)
    assert issubclass(hpr.DecompressionBombWarning, RuntimeWarning)


def test_bomb_rejected_from_header(clip):
    # 20000 x 20000 = 4e8 pixels > 2 * 89,478,485: rejected before any pixel is decoded.
    with pytest.raises(hpr.DecompressionBombError, match="exceeds limit"):
        clip(png_header_only(20000, 20000))


def small():
    # Intermediate buffers are capped at twice the limit too: keep the output small.
    return hpr.AutoImageProcessor.from_dict({"image_processor_type": "CLIPImageProcessor", "size": {"shortest_edge": 32},
                                             "crop_size": {"height": 32, "width": 32}})


@pytest.mark.parametrize("make", [jpeg_bytes, png_bytes])
def test_warning_and_error_thresholds(limit, make, tmp_path):
    clip = small()
    limit(1000)
    assert hpr.get_max_image_pixels() == 1000
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        clip(make(31, 32))  # 992 pixels: fine
    with pytest.warns(hpr.DecompressionBombWarning):
        out = clip(make(40, 40))  # 1600 pixels: > limit, <= 2 * limit
    assert out.shape == (1, 3, 32, 32)
    with pytest.raises(hpr.DecompressionBombError):
        clip(make(45, 45))  # 2025 pixels > 2 * limit
    path = tmp_path / "img.bin"
    path.write_bytes(make(45, 45))
    with pytest.raises(hpr.DecompressionBombError):
        clip(str(path))
    with pytest.raises(hpr.DecompressionBombError):
        hpr.load_image_array(str(path))
    # The default processor upsamples to 224 x 224 (> 2000 pixels): refused under this limit.
    with pytest.raises(hpr.DecompressionBombError, match="resize target"):
        hpr.AutoImageProcessor.from_pretrained(GOLDEN / "configs" / "openai_clip-vit-base-patch32.json")(make(20, 20))
    limit(None)
    assert hpr.get_max_image_pixels() is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        clip(make(45, 45))


def test_warning_as_error(limit):
    clip = small()
    limit(1000)
    with warnings.catch_warnings():
        warnings.simplefilter("error", hpr.DecompressionBombWarning)
        with pytest.raises(hpr.DecompressionBombWarning):
            clip(png_bytes(40, 40))


def test_huge_config_sizes_fail_cleanly(clip):
    """Absurd sizes in a config raise instead of trying to allocate them."""
    arr = np.zeros((20, 30, 3), np.uint8)
    for cfg in (
        {"image_processor_type": "CLIPImageProcessor", "size": {"shortest_edge": 4_000_000_000}},
        {"image_processor_type": "CLIPImageProcessor", "crop_size": {"height": 300_000, "width": 300_000}},
        {"image_processor_type": "ViTImageProcessor", "size": {"height": 4_000_000_000, "width": 4_000_000_000}},
        {"image_processor_type": "Qwen2VLImageProcessor", "min_pixels": 10**15, "max_pixels": 10**16},
    ):
        p = hpr.AutoImageProcessor.from_dict(cfg)
        with pytest.raises(hpr.DecompressionBombError):
            p(arr)


@pytest.mark.parametrize("mean,std", [([0.5, 0.5], [0.5, 0.5, 0.5]), ([], [0.5]), ([0.5], [])])
def test_mismatched_mean_std_raise(mean, std):
    p = hpr.AutoImageProcessor.from_dict({"image_processor_type": "CLIPImageProcessor", "image_mean": mean,
                                          "image_std": std})
    with pytest.raises(ValueError):
        p(np.zeros((20, 30, 3), np.uint8))


def test_whisper_odd_n_fft_shape():
    """Odd n_fft: transformers keeps (n - 1) // hop frames (this used to panic)."""
    fe = hpr.AutoFeatureExtractor.from_dict({"feature_extractor_type": "WhisperFeatureExtractor", "n_fft": 401,
                                             "hop_length": 160, "chunk_length": 1})
    out = fe(np.zeros(1600, np.float32), padding="longest", return_attention_mask=True)
    assert out["input_features"].shape == (1, 80, 9)
    assert out["attention_mask"].shape == (1, 10)


@pytest.mark.parametrize("cfg", [{"hop_length": 0}, {"n_fft": 0}, {"sampling_rate": 0}, {"feature_size": 0},
                                 {"chunk_length": 10**12}, {"n_fft": 10**6}])
def test_whisper_bad_configs_raise(cfg):
    with pytest.raises(ValueError):
        hpr.AutoFeatureExtractor.from_dict({"feature_extractor_type": "WhisperFeatureExtractor", **cfg})
