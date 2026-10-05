# Third-party notices

hf-processors-rs is licensed under `MIT OR Apache-2.0` (see `LICENSE-MIT` and `LICENSE-APACHE`).
Some parts of the source code are ports of code from other projects. Their licenses and
copyright notices are reproduced below, as those licenses require. The ported files also carry
attribution comments.

| Component | Ported into | Upstream license |
|---|---|---|
| [Pillow](https://github.com/python-pillow/Pillow): `src/libImaging/Resample.c`, `Geometry.c` (resize), `AlphaComposite.c` (alpha compositing), `Convert.c` (`cmyk2rgb`) | `src/image/pil_resize.rs`, `src/image/kernels.rs`, `src/image/ops.rs`, `src/image/buffer.rs` | MIT-CMU (HPND) |
| [PyTorch](https://github.com/pytorch/pytorch): `aten/src/ATen/native/cpu/UpSampleKernel.cpp`, `UpSampleKernelAVXAntialias.h`, `aten/src/ATen/native/UpSample.h` (antialiased uint8 resize) | `src/image/torch_resize.rs`, `src/image/kernels.rs` | BSD-3-Clause |
| [torchvision](https://github.com/pytorch/vision): `transforms.v2.functional` resize / center-crop semantics | `src/image/torch_resize.rs`, `src/image/ops.rs` | BSD-3-Clause |
| [Hugging Face transformers](https://github.com/huggingface/transformers): image processor and `WhisperFeatureExtractor` behaviour, `smart_resize`, mel filterbank | `src/processor.rs`, `src/qwen2_vl.rs`, `src/audio/whisper.rs`, `src/config.rs` | Apache-2.0 |

PyTorch's AVX antialiasing kernel is itself derived from
[Pillow-SIMD](https://github.com/uploadcare/pillow-simd), which uses the same PIL license as
Pillow.

## Dependencies that are not vendored

These crates are downloaded by Cargo; no copy of their source is in this repository. Binaries
you distribute that link them must comply with their licenses.

- **libjpeg-turbo** (through the [`mozjpeg`](https://crates.io/crates/mozjpeg) /
  `mozjpeg-sys` crates, only with the optional `pil-jpeg` feature, which the Python package
  enables by default). The C library is compiled from the crate's sources and linked
  statically. It is covered by the IJG License, the Modified (3-clause) BSD License and the
  zlib License; see <https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/LICENSE.md>.
  Binaries that include it, such as Python wheels built with the default features, should
  ship that notice; the IJG license also asks for the statement "this software is based in
  part on the work of the Independent JPEG Group".
- All other Rust dependencies (`image`, `jpeg-decoder`, `ndarray`, `rustfft`, `serde`, `ureq`,
  `rayon`, `candle-core`, `pyo3`, `numpy`, `wasm-bindgen`, ...) are under permissive licenses
  (MIT, Apache-2.0, BSD, Zlib, ISC and similar). Use `cargo about` or `cargo deny` to list
  them for a given build.

## Test data

The test images, audio clips and configuration files under `golden/` are described, with their
sources and licenses, in [`golden/SOURCES.md`](golden/SOURCES.md).

---

## Pillow

Source: <https://github.com/python-pillow/Pillow/blob/main/LICENSE>

```
The Python Imaging Library (PIL) is

    Copyright © 1997-2011 by Secret Labs AB
    Copyright © 1995-2011 by Fredrik Lundh and contributors

Pillow is the friendly PIL fork. It is

    Copyright © 2010 by Jeffrey 'Alex' Clark and contributors

Like PIL, Pillow is licensed under the open source MIT-CMU License:

By obtaining, using, and/or copying this software and/or its associated
documentation, you agree that you have read, understood, and will comply
with the following terms and conditions:

Permission to use, copy, modify and distribute this software and its
documentation for any purpose and without fee is hereby granted,
provided that the above copyright notice appears in all copies, and that
both that copyright notice and this permission notice appear in supporting
documentation, and that the name of Secret Labs AB or the author not be
used in advertising or publicity pertaining to distribution of the software
without specific, written prior permission.

SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
```

## PyTorch

Source: <https://github.com/pytorch/pytorch/blob/main/LICENSE>

```
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions by Tri Dao:
Copyright (c) 2024 Tri Dao.
All rights reserved.

All contributions by Arm:
Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

## torchvision

Source: <https://github.com/pytorch/vision/blob/main/LICENSE>

```
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Hugging Face transformers

Copyright 2018- The Hugging Face team. Licensed under the Apache License, Version 2.0; the
full text is in `LICENSE-APACHE` and at
<https://github.com/huggingface/transformers/blob/main/LICENSE>.
