# hf-processors-rs

[![CI](https://github.com/Dat-cool-repo/hf-processors-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Dat-cool-repo/hf-processors-rs/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

**Like `tokenizers`, but for images and audio.** hf-processors-rs is a Rust crate that reads
a model's `preprocessor_config.json` and reproduces the Hugging Face `transformers` image and
audio preprocessors **bit-exactly**. It covers CLIP, ViT, SigLIP, ConvNeXt, BLIP, Qwen2-VL /
Qwen2.5-VL / Qwen3-VL and Whisper, and has Python and WASM bindings.

- **Exact.** It matches both transformers v5 backends (torchvision and PIL). Golden tests
  against Python report a max abs diff of `0.0` on every image case.
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
| `Qwen2VLImageProcessor` | Qwen2-VL, Qwen2.5-VL, Qwen2.5-Omni, Qwen3-VL, Qwen3.5 | `pixel_values` (P, C·T·p²), `image_grid_thw` (N, 3) |
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

Nothing is published to crates.io or PyPI yet.

**Rust** (git dependency):

```toml
[dependencies]
hf-processors = { git = "https://github.com/Dat-cool-repo/hf-processors-rs", features = ["pil-jpeg"] }
```

The library is imported as `hf_processors`. It needs a recent stable Rust toolchain (edition
2024).

**Python** (build from source; needs a Rust toolchain and Python 3.9 or newer):

```bash
git clone https://github.com/Dat-cool-repo/hf-processors-rs
cd hf-processors-rs/bindings/python
pip install maturin
maturin build --release --out dist
pip install dist/hf_processors_rs-*.whl      # abi3 wheel, import name: hf_processors_rs
```

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
| JPEG input with the default pure-Rust decoder | 8 | 0.016 to 0.052 (about 15% of uint8 values off by 1 to 3) |
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

**Status: experimental.** It builds for `wasm32-unknown-unknown` in CI without the `hub`,
`rayon` and `pil-jpeg` features, so JPEG goes through the pure-Rust decoder. The
`wasm-release` profile produces an 837 KB module (309 KB gzipped). It has not been run in a
browser or in Node yet, and there is no npm package.

```bash
cargo build --profile wasm-release --target wasm32-unknown-unknown -p hf-processors-wasm
wasm-pack build bindings/wasm --target web   # JS glue
```

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
  hub.rs               Hub config lookup, download and cache
bindings/python/       PyO3 + maturin package `hf_processors_rs` and its tests
bindings/wasm/         wasm-bindgen package
golden/                fixture generators, Hub configs, test images, fixtures, manifests
tests/                 Rust golden tests (golden.rs, golden_extra.rs)
examples/              CLI and benchmark
scripts/               helper scripts for Linux / WSL
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
