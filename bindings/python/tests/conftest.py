import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

GOLDEN = Path(os.environ.get("HF_PROCESSORS_GOLDEN", Path(__file__).resolve().parents[3] / "golden"))


@pytest.fixture(scope="session")
def golden() -> Path:
    if not (GOLDEN / "manifest.json").exists():
        pytest.skip(f"golden fixtures not found at {GOLDEN} (set HF_PROCESSORS_GOLDEN)")
    return GOLDEN


def load_manifest(name="manifest.json"):
    return json.loads((GOLDEN / name).read_text())


def sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def invert_lut(pv: np.ndarray, lut: np.ndarray):
    """float32 CHW pixel_values -> uint8 CHW through the exact per-channel LUT, or None when
    some value is not in the LUT (i.e. the output is not bit-exact)."""
    out = np.empty(pv.shape, np.uint8)
    for c in range(pv.shape[0]):
        keys = lut[c].astype(np.float32).view(np.uint32)
        order = np.argsort(keys)
        sk = keys[order]
        vals = np.ascontiguousarray(pv[c]).view(np.uint32).reshape(-1)
        idx = np.clip(np.searchsorted(sk, vals), 0, 255)
        if not np.array_equal(sk[idx], vals):
            return None
        out[c] = order[idx].astype(np.uint8).reshape(pv.shape[1:])
    return out
