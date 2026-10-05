# Motivation and background

This document collects the research behind hf-processors-rs: the problem it addresses, the
evidence that the problem is real, the existing alternatives as of October 2026, and the
original scope and risks. For usage, see the [README](../README.md).

## The problem

Every Rust (or other non-Python) project that runs a vision or audio model has to rewrite
Hugging Face's preprocessing by hand: resize mode, crop, rescale, normalize, patching, mel
filterbanks. These rewrites drift from the Python reference in small ways (interpolation
kernels, rounding, channel order, alpha handling) and quietly lower accuracy. The model still
runs and the outputs still look plausible, so nobody notices.

Text already has a canonical, language-independent library for this step: `tokenizers`.
Images and audio do not.

## Evidence

- A GitHub code search finds **at least 9 projects** that each reimplement
  `CLIPImageProcessor` or `WhisperFeatureExtractor`: candle (llava example), fastembed-rs,
  ahnlich, mlxcel, fashion-clip-rs, smg, ferrum, Crane and aha. Examples:
  - [candle llava `image_processor.rs`](https://github.com/huggingface/candle/blob/main/candle-examples/examples/llava/image_processor.rs)
  - [smg-project/smg](https://github.com/smg-project/smg)
- [tracel-ai/burn#1027](https://github.com/tracel-ai/burn/issues/1027): Burn maintainers note
  that HF preprocessing "isn't declarative or language-independent".
- More copies keep appearing. `CLIPImageProcessor` ports now also exist in
  chenwanqq/candle-llava, and vLLM's Rust frontend and NVIDIA's ai-dynamo parse
  `preprocessor_config.json` themselves.

## transformers v5 made it worse

transformers v5 changed two defaults that existing ports were written against:

1. **The torchvision backend is now the default.** `CLIPImageProcessor` (and every other
   image processor) is now the torchvision implementation, and `AutoImageProcessor` returns it
   whenever torchvision is installed. The previous PIL/numpy implementation lives on as
   `CLIPImageProcessorPil`. The two differ by 1 or 2 uint8 levels in 0.3 to 1% of values (up
   to about 0.03 in `pixel_values`), because torchvision's antialiased resize quantizes its
   weights differently from Pillow's.
2. **`convert_to_rgb` no longer composites alpha on white.** It is a plain
   `convert("RGB")`.

So a Rust port that carefully mimics PIL, or the 4.x RGBA handling, now silently differs from
what `AutoImageProcessor` produces by default.

## What already exists (October 2026)

The check used the crates.io API and GitHub code search.

- **No general, multi-processor HF preprocessing crate with a golden test suite exists.**
- The closest projects:
  - `dynamo-multimodal` 0.1.0 (NVIDIA ai-dynamo, 2026-09-30). It reproduces the image
    pipelines behind HF `AutoProcessor` bit-exactly for one VLM family (Qwen-style patchify
    and M-RoPE), inside an LLM serving router.
  - `turbospark-vision-io` 0.1.0 (2026-09-20). Qwen3.5 vision preprocessing with a
    fixed-point PIL-bicubic resize.
  - `onnx-genai-preprocess`. Image and audio preprocessing for ONNX GenAI models.
  - `pillow-rs`. A Pillow reimplementation that does not know about HF configs.
  - `mel_spec` and `kaldi-native-fbank`. Audio features only.
  - kornia-rs. Image operations, but no HF config compatibility.
- None of them targets CLIP / ViT / SigLIP / ConvNeXt / BLIP / Qwen2-VL from
  `preprocessor_config.json` across both transformers backends, and none has a shared test
  suite against the Python reference.

## Original scope

The first milestone was:

1. Parse `preprocessor_config.json` into a typed processor.
2. Implement the most used processors: `CLIPImageProcessor`, `SiglipImageProcessor`,
   `ViTImageProcessor`, `WhisperFeatureExtractor`, and one dynamic-resolution VLM processor
   (Qwen2-VL-style patching).
3. Golden tests against Python `transformers` for several models (originally a 1e-5
   tolerance; the image processors ended up bit-exact).
4. A Python wheel exposing `from_pretrained("org/model")` that returns numpy arrays.

All four are done. ConvNeXt and BLIP were added along the way.

## Possible next steps

- More processors, by download count: LLaVA-Next, Idefics, PaliGemma, Wav2Vec2, ViT-hybrid,
  video inputs for Qwen2-VL.
- A C ABI (cbindgen header) for Go, Swift and C++.
- A published npm package for the WASM build.
- Offer it upstream as shared preprocessing for candle, fastembed-rs and Burn.

## Risks identified up front, and how they turned out

- **Matching PIL's resize exactly is the hard part.** It needed a full port of Pillow's
  fixed-point resampler, including the intermediate uint8 rounding between passes and the
  pass order. It also needed a second port of PyTorch's antialiased uint8 kernel for the
  torchvision backend. Both are bit-exact on the golden suite.
- **HF processors change between transformers versions.** The golden fixtures are pinned to
  specific versions of transformers, Pillow, torch and torchvision, recorded in
  `golden/manifest.json`. Pillow's next release changes its pass order; `PassOrder::Adaptive`
  already implements that.
- **Scope creep.** There are dozens of processors. The rule is to add them by Hub download
  count, each with golden tests.
