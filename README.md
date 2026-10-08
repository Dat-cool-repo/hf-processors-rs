# hf-processors-rs

[![CI](https://github.com/Dat-cool-repo/hf-processors-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Dat-cool-repo/hf-processors-rs/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

**Like `tokenizers`, but for images and audio.** hf-processors-rs is a Rust crate that reads
a model's `preprocessor_config.json` and reproduces the Hugging Face `transformers` image and
audio preprocessors **bit-exactly**. It covers CLIP, ViT, SigLIP, ConvNeXt, BLIP, Qwen2-VL /
Qwen2.5-VL / Qwen3-VL and Whisper, and has Python and WASM bindings.

- **Exact.** It matches both transformers v5 backends (torchvision and PIL). Golden tests
  against Python report a max abs diff of `0.0` on every image case (JPEG inputs need the
  `pil-jpeg` decoder for that). On 200 real photos, CLIP and Qwen2-VL outputs are
  byte-identical to transformers, and CLIP embeddings computed with candle agree with
  transformers' to a cosine similarity of 0.99999999998 (see [Testing](#testing)).
- **Drop-in.** It reads the same `preprocessor_config.json`, loads it from the Hub, a local
  directory or a file, and returns the same `pixel_values` / `image_grid_thw` /
  `input_features`.
- **Fast.** It is 1.1 to 1.8× faster than transformers single-threaded on decoded images. Batches
  run in parallel without the GIL, and from file paths it is up to 10× faster.

## Why

Every Rust ML stack (candle examples, fastembed-rs, embedded VLM servers...) re-implements
image preprocessing by hand: resize, crop, rescale, normalize, patching, mel filterbanks.
These ports are almost always *subtly* wrong. They use a different bicubic kernel, different
rounding, no antialiasing, or different alpha handling. The model still runs, but its inputs
are not the ones it was trained and evaluated with, and accuracy drops without anyone
noticing.

**transformers v5 made this worse.** `CLIPImageProcessor` and friends now default to the
**torchvision** backend: `AutoImageProcessor` picks it whenever torchvision is installed. The
old PIL implementation became `CLIPImageProcessorPil`. The two differ by 1 or 2 uint8 levels in
0.3 to 1% of values. v5 also stopped compositing RGBA images on white. A Rust port that
carefully mimics PIL, or the 4.x behaviour, now silently differs from what transformers
produces by default.

hf-processors-rs implements **both** backends exactly. Each processor is checked against real
transformers output, so you can pick the one your model was evaluated with. See
[docs/MOTIVATION.md](docs/MOTIVATION.md) for the background research and the survey of
existing crates.

## Supported processors

| transformers class | Models (examples) | Output |
|---|---|---|
| `CLIPImageProcessor` | `openai/clip-vit-base-patch32`, `openai/clip-vit-large-patch14-336` | `pixel_values` (N, 3, H, W) |
| `ViTImageProcessor` | `google/vit-base-patch16-224` | `pixel_values` |
| `SiglipImageProcessor` | `google/siglip-base-patch16-224`, `google/siglip-so400m-patch14-384` | `pixel_values` |
| `ConvNextImageProcessor` | `facebook/convnext-tiny-224` (`crop_pct` path), `facebook/convnext-base-384` (warp path) | `pixel_values` |
| `BlipImageProcessor` | `Salesforce/blip-image-captioning-base` | `pixel_values` |
| `Qwen2VLImageProcessor` | Qwen2-VL, Qwen2.5-VL, Qwen3-VL (golden-tested); other `model_type`s that use this class (Qwen2.5-Omni, Qwen3.5, ...) are mapped to it but not separately tested | `pixel_values` (P, C·T·p²), `image_grid_thw` (N, 3) |
| `WhisperFeatureExtractor` | `openai/whisper-*` | `input_features` (N, 80, 3000), optional `attention_mask` |

Every image processor supports both backends: `torchvision` (the v5 default, `XImageProcessor`)
and `pil` (`XImageProcessorPil`, the v4 "slow" class). Config handling follows transformers:
- the `size` forms `int`, list, `shortest_edge`, `shortest_edge` + `longest_edge`, `height` /
  `width` and `max_height` / `max_width`;
- `get_size_dict` defaults per class, all `do_*` flags, all resample filters;
- the `...FeatureExtractor` / `Fast` / `Pil` class aliases;
- processor-type inference from `config.json`'s `model_type` when the processor config does
  not name one.

## Install

**Python:**

```bash
pip install hf-processors-rs       # import name: hf_processors_rs
```

Prebuilt abi3 wheels (CPython 3.9+) cover Linux x86_64 / aarch64, macOS x86_64 / arm64 and
Windows x64.

**Rust** (not on crates.io yet; git dependency):

```toml
[dependencies]
hf-processors = { git = "https://github.com/Dat-cool-repo/hf-processors-rs", features = ["pil-jpeg"] }
```

The library is imported as `hf_processors`. It needs a recent stable Rust toolchain (edition
2024).

**Python from source** (needs a Rust toolchain, a C compiler for the bundled libjpeg-turbo, and
Python 3.9 or newer):

```bash
git clone https://github.com/Dat-cool-repo/hf-processors-rs
cd hf-processors-rs/bindings/python
python3 -m venv .venv && . .venv/bin/activate   # Windows: py -m venv .venv && .venv\Scripts\activate
pip install maturin
maturin build --release --out dist
pip install dist/hf_processors_rs-*.whl         # abi3 wheel, import name: hf_processors_rs
python -c "import hf_processors_rs as h; print(h.__version__, h.PIL_JPEG)"
```

`pip install .` (from `bindings/python`) also works; it builds the same wheel.

The `Wheels` GitHub workflow builds abi3 wheels for Linux (x86_64, aarch64), Windows x64 and
macOS (x86_64, arm64) as CI artifacts.

## Quick start

### Rust

```rust
use hf_processors::{AutoImageProcessor, AutoProcessor, Backend, Processor, load_image};

fn main() -> hf_processors::Result<()> {
    // A Hub repo id, a local directory, or the path of a preprocessor_config.json.
    let clip = AutoImageProcessor::from_pretrained("openai/clip-vit-base-patch32")?;
    let pixel_values = clip.preprocess_path("cat.jpg")?; // ndarray (3, 224, 224) f32

    let images = vec![load_image("a.png")?, load_image("b.jpg")?];
    let batch = clip.preprocess_batch(&images)?; // (2, 3, 224, 224)

    // Reproduce CLIPImageProcessorPil (the PIL backend) instead of the torchvision default.
    let slow = clip.clone().with_backend(Backend::Pil);

    // Dynamic-resolution VLM processor.
    if let Processor::Qwen2VL(qwen) = AutoProcessor::from_pretrained("Qwen/Qwen2-VL-2B-Instruct")? {
        let out = qwen.preprocess_batch(&images)?;
        // out.pixel_values: (patches, 1176), out.image_grid_thw: (2, 3)
    }

    // Audio: 16 kHz mono f32 samples -> (80, 3000) log-mel features.
    if let Processor::Whisper(fe) = AutoProcessor::from_pretrained("openai/whisper-tiny")? {
        let samples = vec![0.0f32; 16_000];
        let features = fe.extract(&samples)?;
    }
    Ok(())
}
```

`WhisperFeatureExtractor::call` takes `WhisperOptions`, which mirror transformers' call
arguments: `padding`, `max_length`, `truncation`, `pad_to_multiple_of`, `do_normalize`,
`return_attention_mask` and a dither seed. `examples/preprocess.rs` is a small command-line
tool:

```bash
cargo run --release --example preprocess -- openai/clip-vit-base-patch32 golden/images/coffee.png pil
```

### Python

The API mirrors transformers:

```python
import hf_processors_rs as hpr
from PIL import Image

proc = hpr.AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")   # backend="torchvision"
images = [Image.open("cat.jpg"), "dog.png", open("bird.jpg", "rb").read()]       # PIL, paths, bytes, uint8 arrays
pixel_values = proc(images)                         # numpy float32 (3, 3, 224, 224); return_tensors="pt" for torch

slow = hpr.AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32", backend="pil")

qwen = hpr.AutoImageProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
out = qwen(images)                                  # {"pixel_values", "image_grid_thw"}

fe = hpr.AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")
feats = fe(audio, sampling_rate=16000, padding="longest", return_attention_mask=True)
```

- Inputs can be PIL images in any mode (palette, 16-bit and CMYK are handled like Pillow's
  `convert("RGB")`), uint8 numpy arrays (HW, HWC or CHW), torch uint8 tensors, file paths or
  encoded bytes, alone or in lists.
- `num_threads=` sets the size of the rayon pool.
- `from_dict(config)` builds a processor from an in-memory config.

The GIL is released during decoding and preprocessing, which run in parallel over the batch.

Hub downloads use plain HTTPS and fetch only the config files (`preprocessor_config.json`,
and `config.json` when needed). They honour `HF_TOKEN` and `HF_ENDPOINT`, and are cached in
`~/.cache/hf-processors` (`HF_PROCESSORS_CACHE` overrides the location).

## Decompression bombs and size limits

Encoded images (paths and bytes, in Rust, Python and WASM) are checked against a pixel limit
**from the image header, before anything is decoded**, with Pillow's `Image.MAX_IMAGE_PIXELS`
semantics:

| Image size | Behaviour |
|---|---|
| ≤ limit (default 89,478,485 pixels, Pillow's default) | decoded |
| > limit, ≤ 2 × limit | decoded, with a warning (Python: `DecompressionBombWarning`; Rust: stderr, or your handler) |
| > 2 × limit (178,956,970 pixels by default) | `DecompressionBombError` (Python, a `ValueError`) / `Error::DecompressionBomb` (Rust) |

The same hard limit (2 × limit pixels) applies to every buffer the pipeline allocates: resize
targets and the resampler's intermediate image (`input height × output width`), zero-padded
center-crop canvases, Qwen2-VL patch buffers (pixels × temporal patch
size) and Whisper padding and features. A config with an absurd `size`, `crop_size`,
`min_pixels` or `max_length` therefore fails with an error instead of exhausting memory. A
lower limit also caps those buffers, so keep it above your largest output (224 × 224 for CLIP,
`max_pixels` for Qwen2-VL).

```rust
hf_processors::set_max_image_pixels(Some(50_000_000)); // or None to disable, like MAX_IMAGE_PIXELS = None
hf_processors::limits::set_warning_handler(Some(|msg| log::warn!("{msg}"))); // default: one line on stderr
```

```python
import warnings

hpr.set_max_image_pixels(50_000_000)    # hpr.get_max_image_pixels(); None disables the checks
with warnings.catch_warnings():
    warnings.simplefilter("error", hpr.DecompressionBombWarning)   # treat > limit as an error too
    proc("huge.jpg")
```

```js
setMaxImagePixels(50_000_000); // WASM; null / undefined disables the checks
```

The limit is process-wide. NumPy arrays, PIL images and torch tensors are not checked (they
are already decoded; PIL applies its own limit when it opens a file).

Whisper extractor parameters are validated too (`feature_size` 1 to 1024, `n_fft` 1 to 16384,
positive `hop_length` and `sampling_rate`, `chunk_length × sampling_rate` ≤ 2^28 samples), as
are the Qwen2-VL patching parameters (`patch_size` ≤ 1024, `merge_size` and
`temporal_patch_size` ≤ 64). Every released checkpoint is far inside these ranges.

## Backends and exactness

The two backends reproduce two different implementations:

- **`Backend::Torchvision`** (the default) ports ATen's antialiased uint8 resize
  (`upsample_avx_bilinear_bicubic_uint8`: int16 weights with a per-axis precision), the
  torchvision-style center crop, and torch's fused rescale + normalize float math.
- **`Backend::Pil`** ports Pillow's `Resample.c` (22-bit fixed point, horizontal then vertical
  pass, uint8 rounding between them) for every filter, numpy's center crop with zero padding,
  and numpy's rescale (f64 → f32) and normalize (f32) semantics.

Measured against transformers 5.18.0, Pillow 12.3.0, torch 2.14.1 (CPU) and torchvision
0.29.1. The versions are pinned in `golden/manifest.json`.

| What | Cases | Max abs diff vs transformers |
|---|---:|---|
| Image processors, PNG inputs (8 configs × 2 backends, 15 images: L / LA / RGBA / P modes, 1×1 to 3000×2000, extreme aspect ratios) | 142 | **0.0** (bit-identical) |
| JPEG input with `pil-jpeg` (or PIL input from Python) | 8 | **0.0** |
| JPEG input with the default pure-Rust decoder | 8 | 0.016 to 0.052 (8 to 10% of uint8 values off by 1 to 3) |
| Resample overrides (bicubic is the configs' default): nearest, bilinear and lanczos on torchvision; nearest, box, hamming, lanczos and bilinear on PIL; up- and down-sampling | 48 | **0.0** |
| 16-bit PNGs (I;16, RGB16, RGBA16, LA16) | 18 | **0.0** |
| CMYK JPEG | 4 | **0.0** with `pil-jpeg`; ≤ 0.03 without |
| Qwen2-VL / 2.5-VL / 3-VL, plus a 256 to 1280 token budget (both backends, single images and batches) | 92 | **0.0** (`pixel_values`, `image_grid_thw` and shapes identical) |
| Legacy (4.x) alpha compositing on white | 4 | **0.0** |
| Error behaviour (box / hamming on torchvision, Qwen aspect ratio > 200, RGBA without `do_convert_rgb`...) | 40 | Rust returns `Err` wherever transformers raises |
| Whisper log-mel, default call (1 s, 7.3 s, 31 s) | 3 | 1.8e-5 vs torch's float32 STFT; 1.2e-7 vs transformers' numpy path |
| Whisper options (`longest`, `do_normalize`, truncation, `pad_to_multiple_of`, `do_not_pad`, attention masks, seeded dither) | 10 | 2.9e-5; mask shapes and lengths identical |

The image fixtures store SHA-256 hashes of the reference tensors plus the backend's exact
rescale/normalize lookup table, so "0.0" means the float32 bytes are identical. The Python
package runs the same fixtures through its own API, and also compares live against
transformers when transformers and torchvision are installed.

## Feature flags

| Feature | Default | What it does |
|---|:---:|---|
| `hub` | yes | `from_pretrained("org/model")` over HTTPS (`ureq` + rustls). |
| `decode` | yes | `load_image` / `preprocess_path`. PNG uses the `image` crate; JPEG uses the pure-Rust `jpeg-decoder` (within ±3 of Pillow). 16-bit PNGs and CMYK JPEGs are converted the way Pillow converts them. |
| `pil-jpeg` | no | Decodes JPEG with libjpeg-turbo (`mozjpeg-sys`, built from C with `cc`), so pixels are **identical to Pillow**. Needs a C compiler. The Python package enables it by default. |
| `rayon` | no | Parallel batch preprocessing. |
| `candle` | no | `processor::to_candle` converts outputs into `candle_core::Tensor`. |

## Benchmarks

Machine: Intel i9-13900H, WSL2 (shared machine, so expect ±15% noise). The table shows
batch preprocessing of already-decoded images, timing only the processor call (median).
"Typical" is 64 images of about 0.3 MP; "6MP" is 8 images at 3000×2000. The Rust output is
bit-identical to the matching transformers backend. Run it with `bash scripts/bench.sh 1 4`.
These numbers were measured on 2026-10-04 and have not been re-measured since the pre-release
hardening (which adds a header check per decoded image and a few size checks per resize; the
`pil-jpeg` decoder the benchmarks use is unchanged).

**Rust core vs transformers, each called from its own language:**

| Config / batch | Rust tv, 1 thread | transformers tv, 1 thread | transformers PIL, 1 thread | Rust tv, 4 threads | transformers tv, 4 threads |
|---|---:|---:|---:|---:|---:|
| CLIP B/32, typical | **45 ms** | 72 ms | 190 ms | **17 ms** | 56 ms |
| CLIP B/32, 6MP | **49 ms** | 88 ms | 335 ms | **18 ms** | 83 ms |
| ViT-B/16, typical | **52 ms** | 59 ms | 150 ms | **14 ms** | 44 ms |
| ViT-B/16, 6MP | **49 ms** | 82 ms | 209 ms | **12 ms** | 74 ms |
| SigLIP so400m 384, typical | **106 ms** | 134 ms | 349 ms | **47 ms** | 112 ms |
| SigLIP so400m 384, 6MP | **54 ms** | 92 ms | 336 ms | **26 ms** | 95 ms |

The PIL-exact Rust backend takes 61 to 123 ms single-threaded, 2 to 4× faster than the
transformers PIL classes it matches bit-for-bit. The torchvision resize uses an AVX2
`_mm256_madd_epi16` kernel with runtime CPU detection and a scalar fallback. The fallback is
just as exact.

**From Python** (`bindings/python/bench_vs_transformers.py`). Both libraries are called from
Python on PIL images. Our timings include the PIL → numpy conversion.

| Threads | Config / batch | transformers tv | transformers PIL | hf_processors_rs tv | hf_processors_rs PIL | Speed-up vs tv |
|---:|---|---:|---:|---:|---:|---:|
| 1 | CLIP B/32, 64 × 0.3MP | 64 ms | 220 ms | 57 ms | 67 ms | 1.1× |
| 1 | CLIP B/32, 8 × 6MP | 86 ms | 325 ms | 111 ms | 133 ms | 0.8× |
| 1 | CLIP B/32, 64 files (decode included) | 1240 ms | – | 437 ms | – | **2.8×** |
| 1 | SigLIP so400m, 64 × 0.3MP | 122 ms | 331 ms | 80 ms | 89 ms | 1.5× |
| 1 | Qwen2-VL-2B, 64 × 0.3MP | 244 ms | 484 ms | 182 ms | 174 ms | 1.3× |
| 4 | CLIP B/32, 64 × 0.3MP | 47 ms | 196 ms | 30 ms | 32 ms | 1.6× |
| 4 | CLIP B/32, 8 × 6MP | 87 ms | 328 ms | 74 ms | 79 ms | 1.2× |
| 4 | CLIP B/32, 64 files | 1393 ms | – | 137 ms | – | **10.2×** |
| 4 | SigLIP so400m, 64 × 0.3MP | 95 ms | 367 ms | 49 ms | 59 ms | 2.0× |
| 4 | Qwen2-VL-2B, 64 × 0.3MP | 176 ms | 602 ms | 122 ms | 112 ms | 1.4× |

Pillow's `np.asarray` dominates the 6MP rows: it takes 53 ms for 8 images and holds the GIL.
Passing numpy arrays brings that batch down to 61 ms (1 thread) and 28 ms (4 threads), versus
86 ms for transformers. File paths are fastest, because decoding also runs in parallel in Rust.

## Known divergences and limitations

- **Palette PNGs without `do_convert_rgb`** (ViT, ConvNeXt). transformers feeds the palette
  *indices* into the pipeline and then raises. hf-processors-rs expands the palette instead.
  The result equals transformers on `img.convert("RGB")` bit-exactly, in both backends.
- **CMYK JPEGs without `do_convert_rgb`.** CMYK is converted to RGB at decode time with
  Pillow's formula, so ViT and ConvNeXt accept an image that transformers rejects as a
  4-channel array. With `do_convert_rgb` (CLIP, SigLIP, BLIP, Qwen) the output is bit-exact.
- **Whisper is close, not bit-exact.** The spectrogram is computed in f64; transformers' torch
  path uses a float32 STFT. The max diff is about 3e-5. The seeded dither generator is not
  torch's `randn` stream; to replay a specific torch draw, pass the noise explicitly
  (`call_with_noise` / `noise=`). There is no audio resampling: the input must already be at
  the extractor's sampling rate.
- **The default JPEG decoder is not Pillow's.** Enable `pil-jpeg` for identical pixels.
- **EXIF orientation is not applied**, the same as `PIL.Image.open`.
- **Arithmetic-coded JPEGs are rejected** (both decoders are built without arithmetic
  decoding); Pillow decodes them. They are rare in practice.
- **Image formats:** PNG and JPEG only (Pillow opens many more; pass those as PIL images or
  arrays).
- **Not covered yet:**
  - video inputs for Qwen2-VL (images only, `grid_t = 1`);
  - Pillow's upcoming adaptive pass order (`PassOrder::Adaptive` is implemented but not
    golden-tested, since no Pillow release has it yet);
  - more processor families (LLaVA-Next, Idefics, PaliGemma, Wav2Vec2...);
  - a C ABI.

## WASM

`bindings/wasm` exposes a `Preprocessor` class through `wasm-bindgen`. You pass the config
JSON, then call `preprocessRgba` (canvas `ImageData`), `preprocessPixels`, `preprocessEncoded`
(PNG / JPEG bytes) or `extractAudio`:

```js
const p = new Preprocessor(await (await fetch("preprocessor_config.json")).text(), "torchvision");
const { data, width, height } = ctx.getImageData(0, 0, w, h);
const out = p.preprocessRgba(data, width, height); // out.data: Float32Array, out.shape: [1, 3, 224, 224]
```

**Status: experimental, tested in Node.js.** It builds without the `hub`, `rayon` and
`pil-jpeg` features, so JPEG goes through the pure-Rust decoder (the same pixels as a native
build without `pil-jpeg`, on every architecture). CI runs the module in Node.js on PNG and JPEG
test images with five configs and both backends, plus Whisper, and requires every output to be
byte-identical to the native build (`bindings/wasm/tests/node_check.mjs`). It has not been run
in a browser yet, and there is no npm package. The `wasm-release` profile produces an 852 KB
module (314 KB gzipped); `wasm-pack build` gives 855 KB (345 KB gzipped).

Build it with [wasm-pack](https://github.com/wasm-bindgen/wasm-pack) (`cargo install wasm-pack`), from
the repository root:

```bash
rustup target add wasm32-unknown-unknown
wasm-pack build bindings/wasm --target web       # browsers: bindings/wasm/pkg/
wasm-pack build bindings/wasm --target nodejs --out-dir pkg-node   # Node.js

# Run it in Node.js and compare with the native build:
imgs="golden/images/coffee.png golden/images/photo_500x375.jpg"
cargo run --release -p hf-processors-wasm --example native_reference -- /tmp/hfp-ref golden $imgs
node bindings/wasm/tests/node_check.mjs bindings/wasm/pkg-node /tmp/hfp-ref golden $imgs
```

`cargo build --profile wasm-release --target wasm32-unknown-unknown -p hf-processors-wasm`
builds just the module (no JS glue), optimized for size.

## Testing

**Golden tests** (`tests/golden.rs`, `tests/golden_extra.rs`, and the same fixtures through the
Python API in `bindings/python/tests`), against transformers 5.18.0 / Pillow 12.3.0 / torch
2.14.1 (CPU) / torchvision 0.29.1:

- 150 image cases (8 configs × 2 backends, 15 PNG and JPEG images) bit-exact, plus 14 cases
  where transformers raises and so does Rust, and 4 known divergences (palette PNGs, see
  [Known divergences](#known-divergences-and-limitations));
- 74 more fixed-size cases (resample overrides, 16-bit PNGs, CMYK JPEG, palettes) bit-exact,
  18 expected errors reproduced, 2 known divergences (CMYK without `do_convert_rgb`);
- 92 Qwen2-VL / Qwen2.5-VL / Qwen3-VL cases bit-exact (`pixel_values`, `image_grid_thw`,
  single images and batches), 8 expected errors;
- 4 legacy alpha-compositing cases; Whisper: 3 default calls and 10 option combinations
  (tolerances in the table above).

CI runs the Rust tests on Linux x86_64, Windows x64 and macOS arm64, with and without
`pil-jpeg`, and the Python tests on the same three; the Wheels workflow tests every wheel on
Linux x86_64, Windows x64, macOS x86_64 and macOS arm64 with Python 3.9 and 3.13. Locally,
`HF_PROCESSORS_FORCE_SCALAR=1` runs the same suite on the portable (non-AVX2) kernels. A
`python-vs-transformers` CI job also compares live with the pinned transformers stack.

**Real photos, end to end.** 200 photos from Wikimedia Commons (CC0 / public domain, from the
curated "Quality images" category, 194 RGB and 6 grayscale JPEGs, camera originals up to 24 MP
and 1280-px thumbnails; `scripts/real/fetch_photos.py`). Not committed.

| Check | Result |
|---|---|
| CLIP ViT-B/32 `pixel_values`, hf-processors (`pil-jpeg`, torchvision backend) vs `CLIPProcessor` | 200 / 200 byte-identical |
| CLIP image embeddings: [`examples/clip-candle`](examples/clip-candle) (candle, CPU) vs `CLIPModel.get_image_features` (torch, CPU) | max abs diff 1.7e-5 (values up to 8.5), cosine similarity ≥ 0.99999999998 |
| The same torch model on our `pixel_values` vs on transformers' | identical embeddings |
| CLIP `pixel_values`, both backends (`hf_processors_rs` vs `CLIPImageProcessor` / `CLIPImageProcessorPil`) | 200 / 200 identical, each backend |
| Qwen2-VL-2B `pixel_values` + `image_grid_thw`, both backends (3,007,840 patches) | 200 / 200 identical, each backend |

Reproduce with `scripts/real/get_clip.py`, `examples/clip-candle`, `scripts/real/compare_clip.py`
and `scripts/real/compare_processors.py` (each file's docstring has the command).

**Fuzzing** (`fuzz/`, cargo-fuzz / libFuzzer with AddressSanitizer). Six targets:
`config` (arbitrary `preprocessor_config.json` text, or arbitrary field values including NaN /
infinite / zero / negative mean, std and rescale factor, absurd sizes and missing fields, then
every processor class on small images with both backends), `decode_pure` (PNG and the pure-Rust
JPEG decoder), `decode_libjpeg` (libjpeg-turbo's C decoder, compiled with clang's
AddressSanitizer and coverage instrumentation), `resize` (both resamplers, all filters,
arbitrary sizes and channel counts, SIMD kernels checked against the portable ones on every
input), `smart_resize` (extreme sizes, aspect ratios and pixel budgets, plus Qwen2-VL patching)
and `whisper` (arbitrary f32 audio including NaN / inf / empty, call options and extractor
parameters). Each target ran for 20 minutes with two workers (`-jobs=2 -rss_limit_mb=2048
-timeout=10`, the `Fuzz (long)` workflow on GitHub's runners), about 4 CPU-hours and 1.7
million executions in all, ending with no open crash. Fixed findings: Whisper with an odd
`n_fft` panicked (frame-count mismatch); a resize from a tall, narrow image to a wide, short
one allocated a huge intermediate image (timeouts, multi-GB allocations). The size limits and
config validation above were added in the same pass, before the long runs. The minimized
inputs are in `fuzz/regressions/`; CI replays them and fuzzes every target for 30 s on each run.

**Robustness tests** (`tests/robustness.rs`, `bindings/python/tests/test_limits.py`): the
fuzzing regressions, decompression-bomb thresholds, absurd configs, NaN / infinite / empty
audio.

**WASM:** the module runs in Node.js in CI and must match the native build byte for byte
(see [WASM](#wasm)).

**Packaging:** CI checks that every wheel and the sdist carry the license files, runs
`twine check --strict`, and builds and tests the package from the sdist in a fresh venv. The
Windows wheel from the Wheels workflow was also installed into a fresh Python 3.10 venv on a
Windows 11 machine: 433 tests pass (8 skipped: the live transformers comparison).

**Platforms:** Linux x86_64 (CI and local, WSL2), Windows x64 (CI and local), macOS arm64 (CI
and by hand on Apple Silicon, M5 Pro: all Rust golden tests bit-exact, 433 Python tests pass with
both a local build and the CI wheel, and WASM in Node matches native), macOS x86_64 (CI), Linux
aarch64 (wheel built in CI, not tested).

## Development

The golden fixtures are committed, so the Rust and Python test suites need neither
transformers nor network access.

```bash
# Rust: unit + golden tests
cargo test --release --features pil-jpeg         # bit-exact, including JPEG
cargo test --release                             # pure Rust (JPEG checked within bounds)
HF_PROCESSORS_FORCE_SCALAR=1 cargo test --release --features pil-jpeg   # portable non-AVX2 kernels, also bit-exact
cargo clippy --workspace --all-targets --features pil-jpeg,rayon -- -D warnings

# Python package: build the wheel and run the fixture parity tests
cd bindings/python && maturin build --release --out dist \
  && pip install --force-reinstall dist/hf_processors_rs-*.whl && python -m pytest -q tests
```

`scripts/` wraps the same commands for Linux and WSL. `scripts/env.sh` sources an optional,
uncommitted `scripts/env.local.sh` for machine-specific settings such as `CARGO_TARGET_DIR`,
`CARGO_BUILD_JOBS` or `HF_HOME`.

```bash
bash scripts/cargo.sh test --release --features pil-jpeg
bash scripts/setup_venv.sh        # venv with the pinned reference stack (uv, CPU torch)
bash scripts/build_wheel.sh       # build the wheel and install it into that venv
bash scripts/bench.sh 1 4         # Rust vs transformers benchmark
bash scripts/hub_check.sh         # network: from_pretrained on Hub repos, Rust vs Python
```

**Fuzzing** needs nightly Rust and `cargo install cargo-fuzz`. Targets: `config`,
`decode_pure`, `decode_libjpeg`, `resize`, `smart_resize`, `whisper`.

```bash
python fuzz/make_seeds.py                                   # seed corpora (Pillow + numpy)
bash fuzz/run.sh decode_pure 1200 -jobs=2 -workers=2        # 20 minutes, corpora in fuzz/corpus
CLANG=$(command -v clang) bash fuzz/run.sh decode_libjpeg 1200 -jobs=2 -workers=2   # + ASan / coverage in libjpeg-turbo's C code
```

Crashes land in `fuzz/artifacts/<target>/`; once fixed, add the minimized input to
`fuzz/regressions/<target>/` (CI replays them). The `Fuzz (long)` workflow runs every target
for 20 minutes on GitHub's runners.

**Regenerating the golden fixtures** requires the pinned stack: transformers 5.18.0, Pillow
12.3.0, torch 2.14.1 (CPU), torchvision 0.29.1 and numpy 2.5.3.

```bash
bash scripts/setup_venv.sh
bash scripts/py.sh golden/make_golden.py         # suite 1: images, fixtures, manifest.json
bash scripts/py.sh golden/make_golden_extra.py   # suite 2: manifest_extra.json (run after suite 1)
```

To keep fixtures small, most image cases are stored as SHA-256 hashes plus exact lookup
tables. To also keep the full uint8 reference tensors, set `HF_PROCESSORS_GOLDEN_FULL` to a
directory outside the repo when running the generators, and again when running the tests.
Failing cases then report per-pixel differences. When the variable is unset, the tests rely on
the committed hashes, and the per-pixel diagnostics are skipped.

The Python tests read the fixtures from `golden/` by default (`HF_PROCESSORS_GOLDEN`
overrides the location). `test_vs_transformers.py` runs only when transformers and torchvision
are installed.

## Project layout

```
src/
  config.rs            preprocessor_config.json parsing, transformers' size rules
  processor.rs         fixed-size processors (CLIP, ViT, SigLIP, ConvNeXt, BLIP), backends
  qwen2_vl.rs          Qwen2VLImageProcessor: smart_resize, patching, image_grid_thw
  audio/whisper.rs     WhisperFeatureExtractor
  image/pil_resize.rs  Pillow-exact resampler
  image/torch_resize.rs  torchvision / ATen-exact antialiased uint8 resampler
  image/kernels.rs     integer convolution kernels (AVX2 madd, scalar fallback)
  image/ops.rs         RGB conversion, crops, rescale + normalize
  image/buffer.rs      image buffer and Pillow-compatible decoding
  limits.rs            decompression-bomb / buffer-size limit (MAX_IMAGE_PIXELS semantics)
  hub.rs               Hub config lookup, download and cache
bindings/python/       PyO3 + maturin package `hf_processors_rs`, its tests and license files
bindings/wasm/         wasm-bindgen package, native reference + Node.js check
golden/                fixture generators, Hub configs, test images, fixtures, manifests
tests/                 Rust golden tests (golden.rs, golden_extra.rs), robustness.rs (fuzz regressions, limits)
fuzz/                  cargo-fuzz targets, seed generator, committed regression inputs
examples/              CLI, benchmark, clip-candle/ (CLIP embeddings with candle)
scripts/               helper scripts for Linux / WSL; scripts/real/ (real-photo checks)
docs/MOTIVATION.md     background and research
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in this project by you, as defined in the Apache-2.0 license, shall be dual licensed as
above, without any additional terms or conditions.

The resamplers are ports of Pillow (MIT-CMU / HPND) and PyTorch / torchvision
(BSD-3-Clause) code, and the optional `pil-jpeg` feature links libjpeg-turbo. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The test data sources are listed in
[golden/SOURCES.md](golden/SOURCES.md).

## Acknowledgements

- [Hugging Face transformers](https://github.com/huggingface/transformers), the reference
  implementation this project reproduces.
- [Pillow](https://github.com/python-pillow/Pillow) and
  [Pillow-SIMD](https://github.com/uploadcare/pillow-simd), for the resampling algorithms.
- [PyTorch](https://github.com/pytorch/pytorch) and [torchvision](https://github.com/pytorch/vision),
  for the antialiased uint8 resize.
- [libjpeg-turbo](https://libjpeg-turbo.org/), through the
  [`mozjpeg`](https://crates.io/crates/mozjpeg) crate.
- [scikit-image](https://scikit-image.org/), for the public-domain `astronaut` and `coffee`
  test images.
- [PyO3](https://pyo3.rs/), [maturin](https://www.maturin.rs/),
  [rust-numpy](https://github.com/PyO3/rust-numpy),
  [ndarray](https://github.com/rust-ndarray/ndarray), [RustFFT](https://github.com/ejmahler/RustFFT)
  and [wasm-bindgen](https://github.com/rustwasm/wasm-bindgen).
