//! A bit-exact port of PyTorch's antialiased uint8 resize
//! (`F.interpolate(..., antialias=True)` on `uint8` CPU tensors, i.e.
//! `upsample_avx_bilinear_bicubic_uint8` in `ATen/native/cpu/UpSampleKernelAVXAntialias.h`
//! and its generic fallback in `UpSampleKernel.cpp`).
//!
//! This is what `torchvision.transforms.v2.functional.resize` runs for the "fast"
//! (`TorchvisionBackend`) image processors that `transformers` v5 selects by default.
//! It is a Pillow-SIMD derivative: same filters and support as Pillow, but weights are
//! quantised to `i16` with a per-axis precision chosen from the largest weight, so results
//! differ from Pillow by +-1 in some pixels.
//!
//! Attribution: the weight computation and the uint8 loops are a Rust port of PyTorch
//! (`aten/src/ATen/native/cpu/UpSampleKernel.cpp`, `UpSampleKernelAVXAntialias.h`,
//! `UpSample.h`), Copyright (c) 2016- Facebook, Inc. and the PyTorch contributors, used under
//! the BSD-3-Clause license. PyTorch's AVX kernel is itself derived from Pillow-SIMD / Pillow
//! (MIT-CMU / HPND license). See THIRD_PARTY_NOTICES.md.

use super::buffer::ImageU8;
use super::kernels::{self, FixedFilter};
use crate::error::{Error, Result};

/// Torchvision interpolation modes supported on uint8 tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TorchInterp {
    /// `InterpolationMode.NEAREST_EXACT` (what transformers maps `PIL.NEAREST` to).
    NearestExact,
    Bilinear,
    Bicubic,
    /// Lanczos-3 (torchvision >= 0.27).
    Lanczos,
}

impl TorchInterp {
    /// `transformers.image_utils.pil_torch_interpolation_mapping`.
    pub fn from_pil(r: super::pil_resize::Resample) -> Result<TorchInterp> {
        use super::pil_resize::Resample as R;
        Ok(match r {
            R::Nearest => TorchInterp::NearestExact,
            R::Bilinear => TorchInterp::Bilinear,
            R::Bicubic => TorchInterp::Bicubic,
            R::Lanczos => TorchInterp::Lanczos,
            R::Box | R::Hamming => {
                return Err(Error::Config(format!(
                    "resample {r:?} is not supported by torchvision on tensors; use the PIL backend"
                )));
            }
        })
    }

    fn interp_size(self) -> usize {
        match self {
            TorchInterp::NearestExact => 1,
            TorchInterp::Bilinear => 2,
            TorchInterp::Bicubic => 4,
            TorchInterp::Lanczos => 6,
        }
    }

    #[inline]
    fn filter(self, x: f64) -> f64 {
        match self {
            TorchInterp::Bilinear => {
                let x = x.abs();
                if x < 1.0 { 1.0 - x } else { 0.0 }
            }
            TorchInterp::Bicubic => {
                // cubic_convolution1/2 from ATen/native/UpSample.h with A = -0.5 (antialias=True).
                const A: f64 = -0.5;
                let x = x.abs();
                if x < 1.0 {
                    ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
                } else if x < 2.0 {
                    ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
                } else {
                    0.0
                }
            }
            TorchInterp::Lanczos => {
                fn sinc(x: f64) -> f64 {
                    if x == 0.0 {
                        return 1.0;
                    }
                    let x = x * std::f64::consts::PI;
                    x.sin() / x
                }
                let x = x.abs();
                if x < 3.0 { sinc(x) * sinc(x / 3.0) } else { 0.0 }
            }
            TorchInterp::NearestExact => unreachable!(),
        }
    }
}

struct Int16Coeffs {
    ksize: usize,
    bounds: Vec<(usize, usize)>,
    weights: Vec<i16>,
    precision: u32,
}

/// `HelperInterpBase::_compute_index_ranges_int16_weights` with `antialias=True`,
/// `align_corners=False` and no explicit scale.
fn compute_int16_weights(in_size: usize, out_size: usize, mode: TorchInterp) -> Int16Coeffs {
    let scale = in_size as f64 / out_size as f64;
    let half = mode.interp_size() as f64 * 0.5;
    let support = if scale >= 1.0 { half * scale } else { half };
    let ksize = support.ceil() as usize * 2 + 1;
    let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };

    let mut wf = vec![0f64; out_size * ksize];
    let mut bounds = Vec::with_capacity(out_size);
    let mut wt_max = 0f64;
    for i in 0..out_size {
        let center = scale * (i as f64 + 0.5);
        let xmin = ((center - support + 0.5) as i64).max(0);
        let xsize = (((center + support + 0.5) as i64).min(in_size as i64) - xmin).clamp(0, ksize as i64);
        let w = &mut wf[i * ksize..(i + 1) * ksize];
        let mut total = 0f64;
        for (j, wj) in w.iter_mut().enumerate().take(xsize as usize) {
            let v = mode.filter(((j as i64 + xmin) as f64 - center + 0.5) * invscale);
            *wj = v;
            total += v;
        }
        let mut wmax_i = 0f64;
        if total != 0.0 {
            for v in w.iter_mut().take(xsize as usize) {
                *v /= total;
                wmax_i = wmax_i.max(*v);
            }
        }
        wt_max = wt_max.max(wmax_i);
        bounds.push((xmin as usize, xsize as usize));
    }

    let mut precision = 0u32;
    while precision < 22 {
        let next = (0.5 + wt_max * (1i64 << (precision + 1)) as f64) as i32;
        if next >= (1 << 15) {
            break;
        }
        precision += 1;
    }
    let weights = wf
        .iter()
        .map(|&w| {
            let v = w * (1i64 << precision) as f64;
            (if v < 0.0 { (-0.5 + v) as i32 } else { (0.5 + v) as i32 }) as i16
        })
        .collect();
    Int16Coeffs { ksize, bounds, weights, precision }
}

#[cfg(test)]
#[inline(always)]
fn clamp_shift(v: i32, precision: u32) -> u8 {
    (v >> precision).clamp(0, 255) as u8
}

fn horizontal(src: &ImageU8, c: &Int16Coeffs) -> ImageU8 {
    let weights: Vec<i32> = c.weights.iter().map(|&w| w as i32).collect();
    let f = FixedFilter { bounds: &c.bounds, weights: &weights, w16: Some(&c.weights), ksize: c.ksize, precision: c.precision };
    let data = kernels::horizontal(&src.data, src.width, src.channels, 0, src.height, &f);
    ImageU8 { width: c.bounds.len(), height: src.height, channels: src.channels, data }
}

fn vertical(src: &ImageU8, c: &Int16Coeffs) -> ImageU8 {
    let weights: Vec<i32> = c.weights.iter().map(|&w| w as i32).collect();
    let f = FixedFilter { bounds: &c.bounds, weights: &weights, w16: Some(&c.weights), ksize: c.ksize, precision: c.precision };
    let data = kernels::convolve_rows(&src.data, src.width * src.channels, &f);
    ImageU8 { width: src.width, height: c.bounds.len(), channels: src.channels, data }
}

/// Per-pixel port of ATen's `basic_loop_separable_1d_horizontal<uint8_t>` (test reference).
#[cfg(test)]
fn horizontal_reference(src: &ImageU8, c: &Int16Coeffs) -> ImageU8 {
    let ch = src.channels;
    let out_w = c.bounds.len();
    let mut out = ImageU8::zeros(out_w, src.height, ch);
    let init = 1i32 << (c.precision - 1);
    for y in 0..src.height {
        let line = src.row(y);
        let orow = &mut out.data[y * out_w * ch..(y + 1) * out_w * ch];
        for xx in 0..out_w {
            let (xmin, n) = c.bounds[xx];
            let k = &c.weights[xx * c.ksize..xx * c.ksize + n];
            let px = &line[xmin * ch..(xmin + n) * ch];
            for cc in 0..ch {
                let mut s = init;
                for (x, &w) in k.iter().enumerate() {
                    s = s.wrapping_add(px[x * ch + cc] as i32 * w as i32);
                }
                orow[xx * ch + cc] = clamp_shift(s, c.precision);
            }
        }
    }
    out
}

/// Per-pixel port of the vertical uint8 loop (test reference).
#[cfg(test)]
fn vertical_reference(src: &ImageU8, c: &Int16Coeffs) -> ImageU8 {
    let ch = src.channels;
    let out_h = c.bounds.len();
    let row_len = src.width * ch;
    let mut out = ImageU8::zeros(src.width, out_h, ch);
    let init = 1i32 << (c.precision - 1);
    let mut acc = vec![0i32; row_len];
    for yy in 0..out_h {
        let (ymin, n) = c.bounds[yy];
        let k = &c.weights[yy * c.ksize..yy * c.ksize + n];
        acc.iter_mut().for_each(|a| *a = init);
        for (y, &w) in k.iter().enumerate() {
            let line = src.row(ymin + y);
            let w = w as i32;
            for (a, &p) in acc.iter_mut().zip(line.iter()) {
                *a = a.wrapping_add(p as i32 * w);
            }
        }
        let orow = &mut out.data[yy * row_len..(yy + 1) * row_len];
        for (o, &a) in orow.iter_mut().zip(acc.iter()) {
            *o = clamp_shift(a, c.precision);
        }
    }
    out
}

fn nearest_exact(src: &ImageU8, out_w: usize, out_h: usize) -> ImageU8 {
    // nearest_neighbor_exact_compute_source_index: floorf((dst + 0.5) * scale) with a float scale.
    let idx = |dst: usize, in_size: usize, out_size: usize| -> usize {
        let scale = in_size as f32 / out_size as f32;
        let v = ((dst as f64 + 0.5) * scale as f64) as f32;
        (v.floor() as usize).min(in_size - 1)
    };
    let ch = src.channels;
    let xs: Vec<usize> = (0..out_w).map(|x| idx(x, src.width, out_w)).collect();
    let mut out = ImageU8::zeros(out_w, out_h, ch);
    for y in 0..out_h {
        let line = src.row(idx(y, src.height, out_h));
        let orow = &mut out.data[y * out_w * ch..(y + 1) * out_w * ch];
        for (x, &xi) in xs.iter().enumerate() {
            orow[x * ch..(x + 1) * ch].copy_from_slice(&line[xi * ch..(xi + 1) * ch]);
        }
    }
    out
}

/// `torchvision.transforms.v2.functional.resize(uint8_tensor, [out_h, out_w], mode, antialias=True)`.
///
/// Bilinear matches the native uint8 kernel used on CPUs with AVX2 (on other CPUs
/// torchvision resizes bilinear in float32 and rounds, which can differ by +-1).
pub fn resize(src: &ImageU8, out_w: usize, out_h: usize, mode: TorchInterp) -> ImageU8 {
    assert!(out_w > 0 && out_h > 0);
    if out_w == src.width && out_h == src.height {
        return src.clone();
    }
    if mode == TorchInterp::NearestExact {
        return nearest_exact(src, out_w, out_h);
    }
    let mut cur: Option<ImageU8> = None;
    if out_w != src.width {
        let c = compute_int16_weights(src.width, out_w, mode);
        cur = Some(horizontal(src, &c));
    }
    if out_h != src.height {
        let c = compute_int16_weights(src.height, out_h, mode);
        let input = cur.as_ref().unwrap_or(src);
        cur = Some(vertical(input, &c));
    }
    cur.expect("at least one pass")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_kernels_match_reference_loops() {
        let mut state = 42u64;
        let mut rnd = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        for (w, h, ch) in [(37usize, 23usize, 3usize), (5, 64, 1), (64, 5, 3), (13, 9, 4)] {
            let img = ImageU8::new(w, h, ch, (0..w * h * ch).map(|_| (rnd() & 255) as u8).collect()).unwrap();
            for m in [TorchInterp::Bilinear, TorchInterp::Bicubic, TorchInterp::Lanczos] {
                for (ow, oh) in [(11usize, 50usize), (w * 3, 2), (1, 1)] {
                    let c = compute_int16_weights(w, ow, m);
                    assert_eq!(horizontal(&img, &c), horizontal_reference(&img, &c));
                    let c = compute_int16_weights(h, oh, m);
                    assert_eq!(vertical(&img, &c), vertical_reference(&img, &c));
                }
            }
        }
    }
}

#[cfg(test)]
mod profile {
    use super::*;
    use std::time::Instant;

    /// `cargo test --release profile_stages -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn profile_stages() {
        let (w, h, ch) = (3000usize, 2000usize, 3usize);
        let img = ImageU8::new(w, h, ch, (0..w * h * ch).map(|i| (i * 7 % 251) as u8).collect()).unwrap();
        let (ow, oh) = (336usize, 224usize);
        let cw = compute_int16_weights(w, ow, TorchInterp::Bicubic);
        let chh = compute_int16_weights(h, oh, TorchInterp::Bicubic);
        let time = |label: &str, f: &mut dyn FnMut()| {
            f();
            let t = Instant::now();
            for _ in 0..10 {
                f();
            }
            println!("{label:<28} {:>8.2} ms", t.elapsed().as_secs_f64() * 100.0);
        };
        let t = kernels::transpose(&img.data, w, h, ch);
        let weights: Vec<i32> = cw.weights.iter().map(|&x| x as i32).collect();
        let f = FixedFilter { bounds: &cw.bounds, weights: &weights, w16: Some(&cw.weights), ksize: cw.ksize, precision: cw.precision };
        let r = kernels::convolve_rows(&t, h * ch, &f);
        let hor = horizontal(&img, &cw);
        time("transpose in (18 MB)", &mut || drop(kernels::transpose(&img.data, w, h, ch)));
        time("convolve rows (horizontal)", &mut || drop(kernels::convolve_rows(&t, h * ch, &f)));
        time("transpose out", &mut || drop(kernels::transpose(&r, h, ow, ch)));
        time("horizontal total", &mut || drop(horizontal(&img, &cw)));
        time("vertical", &mut || drop(vertical(&hor, &chh)));
        time("resize total", &mut || drop(resize(&img, ow, oh, TorchInterp::Bicubic)));
        println!("taps: horizontal ksize {} vertical ksize {}", cw.ksize, chh.ksize);
    }
}
