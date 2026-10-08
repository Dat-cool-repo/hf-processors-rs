# hf-processors-rs (Python)

Hugging Face `transformers` image and audio preprocessors reproduced **bit-exactly** in Rust:
CLIP, ViT, SigLIP, ConvNeXt, BLIP, Qwen2-VL / Qwen2.5-VL / Qwen3-VL and Whisper. Both
transformers v5 backends are covered (`backend="torchvision"`, the default, and `backend="pil"`),
and decoding plus preprocessing run in Rust with the GIL released, in parallel over the batch.

**Status: alpha.** See the [project README](https://github.com/Dat-cool-repo/hf-processors-rs)
for the exactness tables, tests and benchmarks.

```bash
pip install hf-processors-rs
```

```python
import hf_processors_rs as hpr

proc = hpr.AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
pixel_values = proc(["cat.jpg", "dog.png"])            # numpy float32 (2, 3, 224, 224)

qwen = hpr.AutoImageProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", backend="pil")
out = qwen("cat.jpg")                                 # {"pixel_values", "image_grid_thw"}

fe = hpr.AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")
feats = fe(audio, sampling_rate=16000)["input_features"]   # (1, 80, 3000)
```

Inputs can be PIL images, uint8 numpy arrays, torch uint8 tensors, file paths or encoded
bytes. Paths and bytes are checked against a pixel limit before decoding, with Pillow's
`Image.MAX_IMAGE_PIXELS` semantics (`hpr.set_max_image_pixels(n)` / `None`;
`DecompressionBombWarning` above the limit, `DecompressionBombError` above twice the limit).

## License

`MIT OR Apache-2.0`. The compiled extension statically links libjpeg-turbo (IJG and BSD-3-Clause
licenses) and permissively licensed Rust crates; all their license texts are included in the
distribution (`.dist-info/licenses/`). This software is based in part on the work of the
Independent JPEG Group.
