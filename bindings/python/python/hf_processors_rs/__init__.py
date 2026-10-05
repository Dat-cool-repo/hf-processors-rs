"""Hugging Face ``transformers`` preprocessors reproduced bit-exactly in Rust.

>>> import hf_processors_rs as hpr
>>> proc = hpr.AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")  # backend="torchvision"
>>> pixel_values = proc(images)            # numpy float32 (N, 3, 224, 224)
>>> slow = hpr.AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32", backend="pil")

``images`` may be a PIL image, a uint8 numpy array (HWC, HW or CHW), a file path, encoded
bytes, or a list of those. Decoding (for paths/bytes) and preprocessing run in Rust with the
GIL released, in parallel over the batch (``num_threads=`` picks the pool size; the default is
one thread per core).

``backend="torchvision"`` (default) reproduces transformers v5's default ``XImageProcessor``
classes; ``backend="pil"`` reproduces the ``XImageProcessorPil`` (v4 "slow") classes.

Decompression bombs: paths and bytes are checked against a pixel limit (from the image
header, before decoding) with Pillow's ``Image.MAX_IMAGE_PIXELS`` semantics: above
``get_max_image_pixels()`` (default 89,478,485) a ``DecompressionBombWarning`` is issued, above
twice the limit ``DecompressionBombError`` (a ``ValueError``) is raised. Change it with
``set_max_image_pixels(n)``, or disable it with ``set_max_image_pixels(None)``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional, Sequence, Union

import numpy as np

from ._hf_processors_rs import (
    DEFAULT_MAX_IMAGE_PIXELS,
    PIL_JPEG,
    DecompressionBombError,
    DecompressionBombWarning,
    NativeProcessor,
    __version__,
    get_max_image_pixels,
    load_image_array,
    set_max_image_pixels,
)

__all__ = [
    "AutoImageProcessor",
    "AutoFeatureExtractor",
    "AutoProcessor",
    "ImageProcessor",
    "Qwen2VLImageProcessor",
    "WhisperFeatureExtractor",
    "load_image_array",
    "set_max_image_pixels",
    "get_max_image_pixels",
    "DEFAULT_MAX_IMAGE_PIXELS",
    "DecompressionBombError",
    "DecompressionBombWarning",
    "PIL_JPEG",
    "__version__",
]

_DIRECT_MODES = ("RGB", "L", "RGBA", "LA")


def _pil_to_array(img) -> np.ndarray:
    """PIL image -> uint8 array with the pixels ``transformers`` would see.

    RGB/L/RGBA/LA are passed through (do_convert_rgb then follows ``Image.convert("RGB")``
    in Rust). Palettes are expanded (to RGBA when the palette has transparency), 16/32-bit
    integer modes are clipped to 0..255 like Pillow's ``convert``, everything else (CMYK,
    YCbCr, ...) goes through ``convert("RGB")``.
    """
    mode = img.mode
    if mode not in _DIRECT_MODES:
        if mode == "P":
            img = img.convert("RGBA" if "transparency" in img.info else "RGB")
        elif mode == "PA":
            img = img.convert("RGBA")
        elif mode in ("1", "I", "F") or mode.startswith("I;16"):
            img = img.convert("L")
        else:
            img = img.convert("RGB")
    return np.asarray(img)


def _is_pil(x) -> bool:
    mod = type(x).__module__
    return mod.startswith("PIL.") and hasattr(x, "mode") and hasattr(x, "convert")


def _one_image(x):
    if _is_pil(x):
        # PIL pixels are always HWC: say so, so that e.g. a 3-pixel-high image is not
        # mistaken for channels-first.
        return (_pil_to_array(x), "channels_last")
    if isinstance(x, (bytes, bytearray, memoryview)):
        return bytes(x)
    if isinstance(x, (str, os.PathLike)):
        return os.fspath(x)
    if hasattr(x, "detach") and hasattr(x, "numpy"):  # torch tensor
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.dtype != np.uint8:
        raise TypeError(
            f"image arrays must be uint8 (got {arr.dtype}); hf_processors_rs reproduces the uint8 pipelines "
            "of transformers (PIL images, decoded files)"
        )
    return arr


def _image_list(images) -> list:
    if isinstance(images, (list, tuple)):
        items = list(images)
    elif isinstance(images, np.ndarray) and images.ndim == 4:
        items = list(images)
    elif hasattr(images, "ndim") and hasattr(images, "numpy") and images.ndim == 4:
        items = list(images)
    else:
        items = [images]
    if not items:
        raise ValueError("no images given")
    return [_one_image(x) for x in items]


def _convert(arr: np.ndarray, return_tensors: Optional[str]):
    if return_tensors in (None, "np"):
        return arr
    if return_tensors == "pt":
        import torch

        return torch.from_numpy(arr)
    raise ValueError(f"return_tensors must be None, 'np' or 'pt', got {return_tensors!r}")


class _Base:
    def __init__(self, native: NativeProcessor):
        self._native = native

    @classmethod
    def _load(cls, repo_or_path, backend):
        return NativeProcessor.from_pretrained(os.fspath(repo_or_path), backend)

    @classmethod
    def from_pretrained(cls, repo_or_path: Union[str, os.PathLike], backend: str = "torchvision"):
        """Load from a Hub repo id (``org/name[@revision]``), a directory or a
        ``preprocessor_config.json`` path. Hub downloads honour ``HF_TOKEN`` / ``HF_ENDPOINT``
        and are cached under ``~/.cache/hf-processors`` (``HF_PROCESSORS_CACHE`` overrides)."""
        return _wrap(cls._load(repo_or_path, backend), expected=cls)

    @classmethod
    def from_dict(cls, config: dict, backend: str = "torchvision", processor_type: Optional[str] = None):
        """Build from a config dict (the contents of ``preprocessor_config.json``)."""
        native = NativeProcessor.from_json(json.dumps(config), backend, processor_type)
        return _wrap(native, expected=cls)

    @property
    def class_name(self) -> str:
        """The ``transformers`` class reproduced (torchvision-backend name)."""
        return self._native.class_name

    @property
    def settings(self) -> dict:
        """Resolved preprocessing settings (after class defaults)."""
        return self._native.settings()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._native!r})"


class _ImageBase(_Base):
    @property
    def backend(self) -> str:
        return self._native.backend

    @backend.setter
    def backend(self, value: str):
        self._native.backend = value


class ImageProcessor(_ImageBase):
    """Fixed-size processors: CLIP, ViT, SigLIP, ConvNeXt, BLIP."""

    def __call__(
        self,
        images,
        return_tensors: Optional[str] = None,
        num_threads: Optional[int] = None,
        input_data_format: Optional[str] = None,
    ):
        """Preprocess ``images`` -> ``pixel_values`` of shape ``(N, C, H, W)`` (float32)."""
        out = self._native.preprocess_images(_image_list(images), num_threads, input_data_format)
        return _convert(out, return_tensors)

    def preprocess(self, images, return_tensors: Optional[str] = None, **kwargs) -> dict:
        """Like ``transformers``' ``processor(images)``: ``{"pixel_values": ...}``."""
        return {"pixel_values": self(images, return_tensors=return_tensors, **kwargs)}


class Qwen2VLImageProcessor(_ImageBase):
    """``Qwen2VLImageProcessor`` (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, ...): dynamic resolution."""

    def __call__(
        self,
        images,
        return_tensors: Optional[str] = None,
        num_threads: Optional[int] = None,
        input_data_format: Optional[str] = None,
    ) -> dict:
        """-> ``{"pixel_values": (sum(t*h*w), C*T*P*P) float32, "image_grid_thw": (N, 3) int64}``."""
        pv, grid = self._native.preprocess_qwen(_image_list(images), num_threads, input_data_format)
        return {"pixel_values": _convert(pv, return_tensors), "image_grid_thw": _convert(grid, return_tensors)}

    preprocess = __call__


class WhisperFeatureExtractor(_Base):
    """``WhisperFeatureExtractor``: log-mel features (torch-path semantics of transformers)."""

    def __call__(
        self,
        raw_speech,
        sampling_rate: Optional[int] = None,
        padding: Union[str, bool] = "max_length",
        max_length: Optional[int] = None,
        truncation: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        do_normalize: Optional[bool] = None,
        return_tensors: Optional[str] = None,
        dither_seed: int = 0,
        noise: Optional[np.ndarray] = None,
    ) -> dict:
        """Same arguments and outputs as ``transformers.WhisperFeatureExtractor.__call__``.

        ``dither`` (from the config) uses a seeded generator (``dither_seed``), or pass
        ``noise`` (standard normal, ``N * padded_length`` values) to reproduce a specific
        ``torch.randn`` draw.
        """
        rate = self._native.settings()["sampling_rate"]
        if sampling_rate is not None and sampling_rate != rate:
            raise ValueError(f"the feature extractor expects {rate} Hz audio, got sampling_rate={sampling_rate}")
        if padding is True:
            padding = "longest"
        elif padding is False:
            padding = "do_not_pad"
        is_batched = (isinstance(raw_speech, np.ndarray) and raw_speech.ndim > 1) or (
            isinstance(raw_speech, (list, tuple)) and len(raw_speech) > 0
            and isinstance(raw_speech[0], (np.ndarray, list, tuple))
        )
        if isinstance(raw_speech, np.ndarray) and raw_speech.ndim > 2:
            raise ValueError("only mono-channel audio is supported")
        clips = [np.ascontiguousarray(np.asarray(c, dtype=np.float32).reshape(-1)) for c in raw_speech] \
            if is_batched else [np.ascontiguousarray(np.asarray(raw_speech, dtype=np.float32).reshape(-1))]
        if noise is not None:
            noise = np.ascontiguousarray(np.asarray(noise, dtype=np.float32).reshape(-1))
        feats, mask = self._native.extract_audio(
            clips, padding, max_length, truncation, pad_to_multiple_of, return_attention_mask, do_normalize,
            dither_seed, noise,
        )
        out = {"input_features": _convert(feats, return_tensors)}
        if mask is not None:
            out["attention_mask"] = _convert(mask, return_tensors)
        return out


_KIND_TO_CLASS = {"image": ImageProcessor, "qwen2_vl": Qwen2VLImageProcessor, "whisper": WhisperFeatureExtractor}


def _wrap(native: NativeProcessor, expected):
    cls = _KIND_TO_CLASS[native.kind]
    if expected is AutoImageProcessor and cls is WhisperFeatureExtractor:
        raise ValueError(f"{native.class_name} is an audio feature extractor; use AutoFeatureExtractor")
    if expected in _KIND_TO_CLASS.values() and cls is not expected:
        raise ValueError(f"config is a {native.class_name}, not a {expected.__name__}")
    return cls(native)


class AutoProcessor(_Base):
    """Loads whichever processor the config describes (image, Qwen2-VL or Whisper)."""


class AutoImageProcessor(_Base):
    """``transformers.AutoImageProcessor``: returns an :class:`ImageProcessor` or a
    :class:`Qwen2VLImageProcessor`."""


class AutoFeatureExtractor(_Base):
    """``transformers.AutoFeatureExtractor``: returns a :class:`WhisperFeatureExtractor` (or an
    image processor for image configs, as transformers does)."""
