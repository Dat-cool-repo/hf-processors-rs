//! A bit-exact port of Pillow's `Image.resize` for 8-bit images (`libImaging/Resample.c`,
//! `Geometry.c`'s nearest-neighbour affine scaling).
//!
//! Pillow resamples separably: coefficients are computed in `f64`, converted to 22-bit
//! fixed point, and each pass rounds and clamps back to `u8`. Reproducing that exactly
//! (including the intermediate `u8` rounding between the horizontal and vertical pass and
//! the pass order) is what makes the output identical to `transformers`' "slow" processors.
//!
//! Attribution: the resampling code is a Rust port of Pillow (`src/libImaging/Resample.c`,
//! `Geometry.c`), Copyright (c) 1997-2011 by Secret Labs AB, Copyright (c) 1995-2011 by Fredrik
//! Lundh and contributors, Copyright (c) 2010 by Jeffrey A. Clark and contributors, used under
//! the MIT-CMU (HPND) license. See THIRD_PARTY_NOTICES.md.

use super::buffer::ImageU8;
use super::kernels::{self, FixedFilter};

/// Pillow resampling filters, numbered like `PIL.Image.Resampling`
/// (and `transformers.image_utils.PILImageResampling`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Resample {
    Nearest = 0,
    Lanczos = 1,
    Bilinear = 2,
    Bicubic = 3,
    Box = 4,
    Hamming = 5,
}

impl Resample {
    pub fn from_pil(v: i64) -> Option<Resample> {
        Some(match v {
            0 => Resample::Nearest,
            1 => Resample::Lanczos,
            2 => Resample::Bilinear,
            3 => Resample::Bicubic,
            4 => Resample::Box,
            5 => Resample::Hamming,
            _ => return None,
        })
    }
}

/// Which Pillow release's pass ordering to reproduce.
///
/// Pillow <= 12.3 always runs the horizontal pass first. Pillow's main branch (merged
/// 2026-09-15, PR #9549, expected in Pillow 13) runs the vertical pass first when the height
/// is reduced by more than twice as much as the width. Because each pass rounds to `u8`, the
/// order changes results by +-1 in rare pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PassOrder {
    /// Pillow <= 12.3: horizontal pass, then vertical pass.
    #[default]
    HorizontalFirst,
    /// Pillow >= 13 (main as of 2026-09): adaptive order.
    Adaptive,
}

const PRECISION_BITS: u32 = 32 - 8 - 2;

#[inline]
fn bicubic_filter(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

#[inline]
fn bilinear_filter(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 { 1.0 - x } else { 0.0 }
}

#[inline]
fn box_filter(x: f64) -> f64 {
    if x > -0.5 && x <= 0.5 { 1.0 } else { 0.0 }
}

#[inline]
fn hamming_filter(x: f64) -> f64 {
    let x = x.abs();
    if x == 0.0 {
        return 1.0;
    }
    if x >= 1.0 {
        return 0.0;
    }
    let x = x * std::f64::consts::PI;
    // C: sin(x) / x * (0.54f + 0.46f * cos(x)) -- float literals promoted to double.
    x.sin() / x * (0.54f32 as f64 + 0.46f32 as f64 * x.cos())
}

#[inline]
fn sinc_filter(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let x = x * std::f64::consts::PI;
    x.sin() / x
}

#[inline]
fn lanczos_filter(x: f64) -> f64 {
    if (-3.0..3.0).contains(&x) { sinc_filter(x) * sinc_filter(x / 3.0) } else { 0.0 }
}

fn filter_of(r: Resample) -> (fn(f64) -> f64, f64) {
    match r {
        Resample::Box => (box_filter, 0.5),
        Resample::Bilinear => (bilinear_filter, 1.0),
        Resample::Hamming => (hamming_filter, 1.0),
        Resample::Bicubic => (bicubic_filter, 2.0),
        Resample::Lanczos => (lanczos_filter, 3.0),
        Resample::Nearest => unreachable!("nearest is not a convolution filter"),
    }
}

/// Fixed-point coefficients for one axis.
pub(crate) struct Coeffs {
    pub ksize: usize,
    /// (first source index, number of taps) per output index.
    pub bounds: Vec<(usize, usize)>,
    /// `ksize` fixed-point weights per output index.
    pub kk: Vec<i32>,
}

/// `precompute_coeffs` + `normalize_coeffs_8bpc` from Resample.c.
pub(crate) fn precompute_coeffs(in_size: usize, in0: f32, in1: f32, out_size: usize, resample: Resample) -> Coeffs {
    let (filter, fsupport) = filter_of(resample);
    let scale = (in1 - in0) as f64 / out_size as f64;
    let filterscale = if scale < 1.0 { 1.0 } else { scale };
    let support = fsupport * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let inv_filterscale = 1.0 / filterscale;

    let mut bounds = Vec::with_capacity(out_size);
    let mut kk = vec![0i32; out_size * ksize];
    let mut k = vec![0f64; ksize];
    for xx in 0..out_size {
        let center = in0 as f64 + (xx as f64 + 0.5) * scale;
        // C `(int)` truncates toward zero.
        let mut xmin = (center - support + 0.5) as i64;
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (center + support + 0.5) as i64;
        if xmax > in_size as i64 {
            xmax = in_size as i64;
        }
        let n = (xmax - xmin).max(0) as usize;
        let mut ww = 0.0f64;
        for (x, kx) in k.iter_mut().enumerate().take(n) {
            let w = filter((x as f64 + xmin as f64 - center + 0.5) * inv_filterscale);
            *kx = w;
            ww += w;
        }
        if ww != 0.0 {
            for v in k.iter_mut().take(n) {
                *v /= ww;
            }
        }
        let dst = &mut kk[xx * ksize..xx * ksize + ksize];
        for x in 0..n {
            let v = k[x];
            let scaled = v * (1u32 << PRECISION_BITS) as f64;
            dst[x] = if v < 0.0 { (-0.5 + scaled) as i32 } else { (0.5 + scaled) as i32 };
        }
        bounds.push((xmin as usize, n));
    }
    Coeffs { ksize, bounds, kk }
}

#[cfg(test)]
#[inline(always)]
fn clip8(v: i32) -> u8 {
    (v >> PRECISION_BITS).clamp(0, 255) as u8
}

/// Horizontal pass on rows `[offset, offset + out_h)` of `src` (fast path).
fn resample_horizontal(src: &ImageU8, offset: usize, out_h: usize, c: &Coeffs) -> ImageU8 {
    let f = FixedFilter { bounds: &c.bounds, weights: &c.kk, w16: None, ksize: c.ksize, precision: PRECISION_BITS };
    let data = kernels::horizontal(&src.data, src.width, src.channels, offset, out_h, &f);
    ImageU8 { width: c.bounds.len(), height: out_h, channels: src.channels, data }
}

/// Vertical pass; `c.bounds` are relative to row 0 of `src` (fast path).
fn resample_vertical(src: &ImageU8, c: &Coeffs) -> ImageU8 {
    let f = FixedFilter { bounds: &c.bounds, weights: &c.kk, w16: None, ksize: c.ksize, precision: PRECISION_BITS };
    let data = kernels::convolve_rows(&src.data, src.width * src.channels, &f);
    ImageU8 { width: src.width, height: c.bounds.len(), channels: src.channels, data }
}

/// Straightforward port of `ImagingResampleHorizontal_8bpc` (reference for tests).
#[cfg(test)]
fn resample_horizontal_reference(src: &ImageU8, offset: usize, out_h: usize, c: &Coeffs) -> ImageU8 {
    let ch = src.channels;
    let out_w = c.bounds.len();
    let mut out = ImageU8::zeros(out_w, out_h, ch);
    let half = 1i32 << (PRECISION_BITS - 1);
    for yy in 0..out_h {
        let line = src.row(yy + offset);
        let orow = &mut out.data[yy * out_w * ch..(yy + 1) * out_w * ch];
        for xx in 0..out_w {
            let (xmin, n) = c.bounds[xx];
            let k = &c.kk[xx * c.ksize..xx * c.ksize + n];
            let px = &line[xmin * ch..(xmin + n) * ch];
            match ch {
                3 => {
                    let (mut s0, mut s1, mut s2) = (half, half, half);
                    for (x, &w) in k.iter().enumerate() {
                        s0 = s0.wrapping_add(px[x * 3] as i32 * w);
                        s1 = s1.wrapping_add(px[x * 3 + 1] as i32 * w);
                        s2 = s2.wrapping_add(px[x * 3 + 2] as i32 * w);
                    }
                    orow[xx * 3] = clip8(s0);
                    orow[xx * 3 + 1] = clip8(s1);
                    orow[xx * 3 + 2] = clip8(s2);
                }
                _ => {
                    for cc in 0..ch {
                        let mut s = half;
                        for (x, &w) in k.iter().enumerate() {
                            s = s.wrapping_add(px[x * ch + cc] as i32 * w);
                        }
                        orow[xx * ch + cc] = clip8(s);
                    }
                }
            }
        }
    }
    out
}

/// Straightforward port of `ImagingResampleVertical_8bpc` (reference for tests).
#[cfg(test)]
fn resample_vertical_reference(src: &ImageU8, c: &Coeffs) -> ImageU8 {
    let ch = src.channels;
    let w = src.width;
    let out_h = c.bounds.len();
    let row_len = w * ch;
    let mut out = ImageU8::zeros(w, out_h, ch);
    let half = 1i32 << (PRECISION_BITS - 1);
    let mut acc = vec![0i32; row_len];
    for yy in 0..out_h {
        let (ymin, n) = c.bounds[yy];
        let k = &c.kk[yy * c.ksize..yy * c.ksize + n];
        acc.iter_mut().for_each(|a| *a = half);
        // Integer addition is associative (with wrapping), so accumulating row by row
        // gives exactly the same result as Pillow's per-pixel loop.
        for (y, &wgt) in k.iter().enumerate() {
            let line = src.row(ymin + y);
            for (a, &p) in acc.iter_mut().zip(line.iter()) {
                *a = a.wrapping_add(p as i32 * wgt);
            }
        }
        let orow = &mut out.data[yy * row_len..(yy + 1) * row_len];
        for (o, &a) in orow.iter_mut().zip(acc.iter()) {
            *o = clip8(a);
        }
    }
    out
}

/// Pillow's nearest-neighbour scaling (`ImagingScaleAffine` with a pure scale matrix).
fn resize_nearest(src: &ImageU8, out_w: usize, out_h: usize) -> ImageU8 {
    #[inline]
    fn coord(v: f64) -> i64 {
        if v < 0.0 { -1 } else { v as i64 }
    }
    let a0 = src.width as f32 as f64 / out_w as f64;
    let a4 = src.height as f32 as f64 / out_h as f64;
    let ch = src.channels;
    let mut xin = vec![0usize; out_w];
    let mut xo = a0 * 0.5;
    for v in xin.iter_mut() {
        let xi = coord(xo);
        *v = xi.clamp(0, src.width as i64 - 1) as usize;
        xo += a0;
    }
    let mut out = ImageU8::zeros(out_w, out_h, ch);
    let mut yo = a4 * 0.5;
    for y in 0..out_h {
        let yi = coord(yo).clamp(0, src.height as i64 - 1) as usize;
        yo += a4;
        let line = src.row(yi);
        let orow = &mut out.data[y * out_w * ch..(y + 1) * out_w * ch];
        for (x, &xi) in xin.iter().enumerate() {
            orow[x * ch..(x + 1) * ch].copy_from_slice(&line[xi * ch..(xi + 1) * ch]);
        }
    }
    out
}

/// `PIL.Image.resize((out_w, out_h), resample)` for an 8-bit image with no alpha
/// premultiplication (L or RGB). For LA/RGBA Pillow premultiplies alpha first; the
/// processors in this crate always resize RGB images.
pub fn resize(src: &ImageU8, out_w: usize, out_h: usize, resample: Resample, order: PassOrder) -> ImageU8 {
    assert!(out_w > 0 && out_h > 0, "height and width must be > 0");
    if out_w == src.width && out_h == src.height {
        return src.clone();
    }
    if resample == Resample::Nearest {
        return resize_nearest(src, out_w, out_h);
    }
    let (in_w, in_h) = (src.width, src.height);
    let need_h = out_w != in_w;
    let need_v = out_h != in_h;
    let horizontal_first = match order {
        PassOrder::HorizontalFirst => true,
        PassOrder::Adaptive => {
            let dy = in_h as i64 - out_h as i64;
            let dx = in_w as i64 - out_w as i64;
            !(dy > 0 && dy > dx * 2)
        }
    };
    let cv = precompute_coeffs(in_h, 0.0, in_h as f32, out_h, resample);
    if horizontal_first {
        let mut cur: Option<ImageU8> = None;
        let mut cv = cv;
        if need_h {
            let ch = precompute_coeffs(in_w, 0.0, in_w as f32, out_w, resample);
            let ybox_first = cv.bounds[0].0;
            let last = cv.bounds[out_h - 1];
            let ybox_last = last.0 + last.1;
            for b in cv.bounds.iter_mut() {
                b.0 -= ybox_first;
            }
            cur = Some(resample_horizontal(src, ybox_first, ybox_last - ybox_first, &ch));
        }
        if need_v {
            let input = cur.as_ref().unwrap_or(src);
            cur = Some(resample_vertical(input, &cv));
        }
        cur.unwrap_or_else(|| src.clone())
    } else {
        let mut cur: Option<ImageU8> = None;
        if need_v {
            cur = Some(resample_vertical(src, &cv));
        }
        if need_h {
            let ch = precompute_coeffs(in_w, 0.0, in_w as f32, out_w, resample);
            let input = cur.as_ref().unwrap_or(src);
            let h = input.height;
            cur = Some(resample_horizontal(input, 0, h, &ch));
        }
        cur.unwrap_or_else(|| src.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_size_is_copy() {
        let img = ImageU8::new(3, 2, 3, (0..18).collect()).unwrap();
        assert_eq!(resize(&img, 3, 2, Resample::Bicubic, PassOrder::default()), img);
    }

    #[test]
    fn fast_kernels_match_reference_loops() {
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut rnd = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        for (w, h, ch) in [(37usize, 23usize, 3usize), (5, 64, 1), (64, 5, 3), (13, 9, 4)] {
            let img = ImageU8::new(w, h, ch, (0..w * h * ch).map(|_| (rnd() & 255) as u8).collect()).unwrap();
            for r in [Resample::Bilinear, Resample::Bicubic, Resample::Lanczos, Resample::Box, Resample::Hamming] {
                for (ow, oh) in [(11usize, 50usize), (w * 3, 2), (1, 1)] {
                    let ch_ = precompute_coeffs(w, 0.0, w as f32, ow, r);
                    assert_eq!(resample_horizontal(&img, 0, h, &ch_), resample_horizontal_reference(&img, 0, h, &ch_));
                    let cv = precompute_coeffs(h, 0.0, h as f32, oh, r);
                    assert_eq!(resample_vertical(&img, &cv), resample_vertical_reference(&img, &cv));
                }
            }
        }
    }

    #[test]
    fn constant_image_stays_constant() {
        let img = ImageU8::new(37, 23, 3, vec![200; 37 * 23 * 3]).unwrap();
        for r in [Resample::Bilinear, Resample::Bicubic, Resample::Lanczos, Resample::Box, Resample::Hamming] {
            let out = resize(&img, 11, 50, r, PassOrder::default());
            assert!(out.data.iter().all(|&v| v == 200), "{r:?}");
        }
    }
}
