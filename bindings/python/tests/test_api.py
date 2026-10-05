"""API behaviour: input types, batching, threads, GIL release, errors."""

import threading
import time

import numpy as np
import pytest
from PIL import Image

import hf_processors_rs as hpr
from conftest import GOLDEN


@pytest.fixture(scope="module")
def clip(golden):
    return hpr.AutoImageProcessor.from_pretrained(golden / "configs" / "openai_clip-vit-base-patch32.json")


@pytest.fixture(scope="module")
def photo(golden):
    im = Image.open(golden / "images" / "photo_640x480.png")
    im.load()
    return im


def test_input_types_agree(clip, photo, golden):
    path = golden / "images" / "photo_640x480.png"
    ref = clip(photo)
    arr = np.asarray(photo)
    for x in (str(path), path, path.read_bytes(), arr, np.ascontiguousarray(arr.transpose(2, 0, 1)),
              arr[::1, ::1, :].copy(order="F")):
        np.testing.assert_array_equal(clip(x), ref)
    # A batch mixing all of them, plus a 4-D array.
    out = clip([photo, str(path), arr, path.read_bytes()])
    assert out.shape == (4, 3, 224, 224)
    np.testing.assert_array_equal(out, np.repeat(ref, 4, axis=0))
    np.testing.assert_array_equal(clip(np.stack([arr, arr])), np.repeat(ref, 2, axis=0))


def test_small_pil_images_are_hwc(golden):
    """A PIL image 3 pixels high must not be read as channels-first."""
    q = hpr.AutoImageProcessor.from_pretrained(golden / "configs" / "Qwen_Qwen2-VL-2B-Instruct.json")
    im = Image.open(golden / "images" / "tiny_2x3.png")  # 2 wide, 3 high
    assert q(im)["image_grid_thw"].tolist() == [[1, 6, 4]]


def test_backend_switch(clip, photo, golden):
    pil = hpr.AutoImageProcessor.from_pretrained(golden / "configs" / "openai_clip-vit-base-patch32.json",
                                                 backend="pil")
    assert pil.backend == "pil" and clip.backend == "torchvision"
    a, b = clip(photo), pil(photo)
    assert 0 < np.abs(a - b).max() < 0.05  # the two transformers backends differ by +-1-2 levels
    clip.backend = "pil"
    try:
        np.testing.assert_array_equal(clip(photo), b)
    finally:
        clip.backend = "torchvision"
    with pytest.raises(ValueError):
        clip.backend = "opencv"


def test_num_threads_and_return_tensors(clip, photo):
    batch = [photo] * 16
    ref = clip(batch, num_threads=1)
    for n in (2, 4, None):
        np.testing.assert_array_equal(clip(batch, num_threads=n), ref)
    assert clip.preprocess(photo)["pixel_values"].shape == (1, 3, 224, 224)
    torch = pytest.importorskip("torch")
    t = clip(photo, return_tensors="pt")
    assert isinstance(t, torch.Tensor) and t.dtype == torch.float32


def test_gil_is_released(clip, golden):
    """Another Python thread keeps running while a large batch is processed."""
    big = np.asarray(Image.open(golden / "images" / "huge_3000x2000.png"))
    batch = [big] * 8
    done = threading.Event()
    ticks = 0

    def work():
        clip(batch, num_threads=1)
        done.set()

    t = threading.Thread(target=work)
    start = last = time.perf_counter()
    max_gap = 0.0
    t.start()
    while not done.is_set():  # busy loop: needs the GIL for every iteration
        now = time.perf_counter()
        max_gap = max(max_gap, now - last)
        last = now
        ticks += 1
    t.join()
    elapsed = time.perf_counter() - start
    # Holding the GIL for the native call would stall this loop for (almost) the whole call.
    assert elapsed < 0.02 or max_gap < elapsed / 2, (elapsed, max_gap, ticks)


def test_errors(clip, golden):
    with pytest.raises(TypeError):
        clip(np.zeros((10, 10, 3), np.float32))
    with pytest.raises(ValueError):
        clip(np.zeros((10, 10, 7), np.uint8))
    with pytest.raises(OSError):
        clip(str(golden / "images" / "does_not_exist.png"))
    with pytest.raises(ValueError):
        clip(b"not an image")
    with pytest.raises(ValueError):
        clip([])
    with pytest.raises(ValueError, match="AutoFeatureExtractor"):
        hpr.AutoImageProcessor.from_pretrained(golden / "configs" / "openai_whisper-tiny.json")
    with pytest.raises(ValueError):
        hpr.AutoProcessor.from_dict({"image_processor_type": "LlavaNextImageProcessor"})


def test_settings_and_repr(clip):
    s = clip.settings
    assert s["crop_size"] == (224, 224) and s["do_convert_rgb"] is True
    assert clip.class_name == "CLIPImageProcessor"
    assert "CLIPImageProcessor" in repr(clip)


def test_whisper_inputs(golden):
    fe = hpr.AutoFeatureExtractor.from_pretrained(golden / "configs" / "openai_whisper-tiny.json")
    x = (np.sin(np.arange(16000) * 0.05) * 0.3).astype(np.float64)  # float64 is cast like transformers
    a = fe(x, sampling_rate=16000)["input_features"]
    b = fe([x.tolist()], sampling_rate=16000)["input_features"]
    c = fe(np.stack([x, x]), sampling_rate=16000)["input_features"]
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(np.repeat(a, 2, axis=0), c)
    with pytest.raises(ValueError):
        fe(x, sampling_rate=8000)
    out = fe([x, x[:8000]], sampling_rate=16000, padding=True, return_attention_mask=True)
    assert out["input_features"].shape == (2, 80, 100)
    assert out["attention_mask"].sum(-1).tolist() == [100, 50]
