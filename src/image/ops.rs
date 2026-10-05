//! Pixel operations other than resizing: RGB conversion, center crop, rescale + normalize.
//!
//! Attribution: the alpha compositor is a Rust port of Pillow
//! (`src/libImaging/AlphaComposite.c`), Copyright (c) 1997-2011 by Secret Labs AB, Copyright
//! (c) 1995-2011 by Fredrik Lundh and contributors, Copyright (c) 2010 by Jeffrey A. Clark and
//! contributors, used under the MIT-CMU (HPND) license. See THIRD_PARTY_NOTICES.md.

use super::buffer::ImageU8;
use ndarray::Array3;

/// Pillow's `SHIFTFORDIV255`: a rounded division by 255 for the alpha compositor.
#[inline]
fn shift_for_div255(a: u32) -> u32 {
    ((a >> 8) + a) >> 8
}

/// `Image.alpha_composite(white, src)` for one RGBA pixel, exactly as in
/// Pillow's `libImaging/AlphaComposite.c` (7 extra bits of precision).
#[inline]
fn composite_on_white(r: u8, g: u8, b: u8, a: u8) -> [u8; 3] {
    const PB: u32 = 7;
    if a == 0 {
        return [255, 255, 255];
    }
    let (sa, da) = (a as u32, 255u32);
    let blend = da * (255 - sa);
    let outa255 = sa * 255 + blend;
    let coef1 = sa * 255 * 255 * (1 << PB) / outa255;
    let coef2 = 255 * (1 << PB) - coef1;
    let ch = |s: u8| -> u8 {
        let tmp = s as u32 * coef1 + 255 * coef2;
        (shift_for_div255(tmp + (0x80 << PB)) >> PB) as u8
    };
    [ch(r), ch(g), ch(b)]
}

/// Legacy `transformers` (4.4x) `convert_to_rgb`: every non-RGB image is converted to RGBA
/// and alpha-composited on an opaque white background (`Image.alpha_composite`), then
/// converted to RGB. Bit-exact port of Pillow's integer compositor.
pub fn convert_to_rgb_composite(img: &ImageU8) -> ImageU8 {
    let n = img.width * img.height;
    let mut data = Vec::with_capacity(n * 3);
    match img.channels {
        3 => return img.clone(),
        1 => {
            // L -> RGBA has alpha 255; compositing an opaque pixel is the identity
            // (verified exhaustively in tests), so this is just replication.
            for &l in &img.data {
                data.extend_from_slice(&[l, l, l]);
            }
        }
        2 => {
            for p in img.data.as_chunks::<2>().0 {
                data.extend_from_slice(&composite_on_white(p[0], p[0], p[0], p[1]));
            }
        }
        4 => {
            for p in img.data.as_chunks::<4>().0 {
                data.extend_from_slice(&composite_on_white(p[0], p[1], p[2], p[3]));
            }
        }
        c => unreachable!("channels={c}"),
    }
    ImageU8 { width: img.width, height: img.height, channels: 3, data }
}

/// `transformers.image_transforms.convert_to_rgb` as of transformers v5: a plain
/// `PIL.Image.convert("RGB")` (alpha dropped, L/LA replicated, palettes expanded).
pub fn convert_to_rgb(img: &ImageU8) -> ImageU8 {
    let mut data = Vec::with_capacity(img.width * img.height * 3);
    match img.channels {
        3 => return img.clone(),
        1 => img.data.iter().for_each(|&l| data.extend_from_slice(&[l, l, l])),
        2 => img.data.as_chunks::<2>().0.iter().for_each(|p| data.extend_from_slice(&[p[0], p[0], p[0]])),
        4 => img.data.as_chunks::<4>().0.iter().for_each(|p| data.extend_from_slice(&p[..3])),
        c => unreachable!("channels={c}"),
    }
    ImageU8 { width: img.width, height: img.height, channels: 3, data }
}

/// Floor division as in Python's `//`.
#[inline]
fn floor_div2(v: i64) -> i64 {
    v.div_euclid(2)
}

/// `transformers.image_transforms.center_crop` (numpy, "slow" processors): crops with
/// `top = (h - crop_h) // 2` and zero-pads when the image is smaller than the crop.
pub fn center_crop_slow(img: &ImageU8, crop_h: usize, crop_w: usize) -> ImageU8 {
    let (h, w) = (img.height as i64, img.width as i64);
    let (ch_, cw_) = (crop_h as i64, crop_w as i64);
    let top = floor_div2(h - ch_);
    let left = floor_div2(w - cw_);
    if top >= 0 && top + ch_ <= h && left >= 0 && left + cw_ <= w {
        return img.crop(top as usize, left as usize, crop_h, crop_w);
    }
    // Pad: place the image in a zero canvas of max(crop, orig) size, then crop.
    let new_h = ch_.max(h);
    let new_w = cw_.max(w);
    let top_pad = (new_h - h + 1) / 2; // ceil((new_h - h) / 2)
    let left_pad = (new_w - w + 1) / 2;
    let c = img.channels;
    let mut canvas = ImageU8::zeros(new_w as usize, new_h as usize, c);
    for y in 0..img.height {
        let dst = ((y as i64 + top_pad) * new_w + left_pad) as usize * c;
        canvas.data[dst..dst + img.width * c].copy_from_slice(img.row(y));
    }
    let (t, b) = (top + top_pad, top + ch_ + top_pad);
    let (l, r) = (left + left_pad, left + cw_ + left_pad);
    let (t, b) = (t.max(0), b.min(new_h));
    let (l, r) = (l.max(0), r.min(new_w));
    canvas.crop(t as usize, l as usize, (b - t) as usize, (r - l) as usize)
}

/// `TorchvisionBackend.center_crop` ("fast" processors, transformers v5): zero padding
/// (`(crop - size) // 2` before, the rest after) when the crop is larger than the image,
/// then `top = int((h - crop_h) / 2.0)`. Note the padding split differs from the numpy
/// version (`ceil` before) for odd differences.
pub fn center_crop_torchvision(img: &ImageU8, crop_h: usize, crop_w: usize) -> ImageU8 {
    let (mut cur, mut h, mut w) = (img.clone(), img.height, img.width);
    if crop_h > h || crop_w > w {
        let pad_l = if crop_w > w { (crop_w - w) / 2 } else { 0 };
        let pad_t = if crop_h > h { (crop_h - h) / 2 } else { 0 };
        let pad_r = if crop_w > w { (crop_w - w).div_ceil(2) } else { 0 };
        let pad_b = if crop_h > h { (crop_h - h).div_ceil(2) } else { 0 };
        let (nw, nh) = (w + pad_l + pad_r, h + pad_t + pad_b);
        let c = img.channels;
        let mut canvas = ImageU8::zeros(nw, nh, c);
        for y in 0..h {
            let dst = ((y + pad_t) * nw + pad_l) * c;
            canvas.data[dst..dst + w * c].copy_from_slice(img.row(y));
        }
        cur = canvas;
        h = nh;
        w = nw;
        if crop_h == h && crop_w == w {
            return cur;
        }
    }
    // transformers' TorchvisionBackend.center_crop: int((h - crop_h) / 2.0), i.e. floor.
    let top = (h - crop_h) / 2;
    let left = (w - crop_w) / 2;
    cur.crop(top, left, crop_h, crop_w)
}

/// Rescale + normalize parameters, applied with the exact float semantics of
/// `transformers.image_transforms.rescale` / `normalize` (slow processors):
/// `x = f32(f64(u8) * rescale_factor)`, then `(x - f32(mean)) / f32(std)` in f32.
#[derive(Clone, Debug)]
pub struct Normalize {
    pub do_rescale: bool,
    pub rescale_factor: f64,
    pub do_normalize: bool,
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

/// Which library's float arithmetic to reproduce for rescale/normalize.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatSemantics {
    /// numpy: rescale in f64 then cast to f32; normalize in f32.
    Numpy,
    /// torch fast processors: mean/std pre-multiplied by `1/rescale_factor` in f32,
    /// then `(f32(u8) - mean') / std'` in f32.
    TorchFused,
}

impl Normalize {
    /// Per-channel lookup tables: output value for every possible u8 input.
    pub fn luts(&self, channels: usize, sem: FloatSemantics) -> Vec<[f32; 256]> {
        (0..channels)
            .map(|c| {
                let mut lut = [0f32; 256];
                let mean = if self.mean.len() == 1 { self.mean[0] } else { self.mean[c] };
                let std = if self.std.len() == 1 { self.std[0] } else { self.std[c] };
                for (v, out) in lut.iter_mut().enumerate() {
                    *out = match sem {
                        FloatSemantics::Numpy => {
                            let x = if self.do_rescale {
                                (v as f64 * self.rescale_factor) as f32
                            } else {
                                v as f32
                            };
                            if self.do_normalize { (x - mean as f32) / std as f32 } else { x }
                        }
                        FloatSemantics::TorchFused => {
                            if self.do_rescale && self.do_normalize {
                                let inv = (1.0 / self.rescale_factor) as f32;
                                let m = mean as f32 * inv;
                                let s = std as f32 * inv;
                                (v as f32 - m) / s
                            } else if self.do_normalize {
                                (v as f32 - mean as f32) / std as f32
                            } else if self.do_rescale {
                                // torch: uint8 tensor * python float -> float32
                                v as f32 * self.rescale_factor as f32
                            } else {
                                v as f32
                            }
                        }
                    };
                }
                lut
            })
            .collect()
    }

    /// Convert an HWC u8 image into a CHW f32 array.
    pub fn apply_chw(&self, img: &ImageU8, sem: FloatSemantics) -> Array3<f32> {
        apply_luts_chw(img, &self.luts(img.channels, sem))
    }
}

/// Map an HWC u8 image through per-channel lookup tables into a CHW f32 array.
pub fn apply_luts_chw(img: &ImageU8, luts: &[[f32; 256]]) -> Array3<f32> {
    let (h, w, c) = (img.height, img.width, img.channels);
    assert_eq!(luts.len(), c);
    let mut out = vec![0f32; c * h * w];
    let plane = h * w;
    for (i, px) in img.data.chunks_exact(c).enumerate() {
        for ch in 0..c {
            out[ch * plane + i] = luts[ch][px[ch] as usize];
        }
    }
    Array3::from_shape_vec((c, h, w), out).expect("shape")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opaque_composite_is_identity() {
        for v in 0..=255u8 {
            assert_eq!(composite_on_white(v, v, v, 255), [v, v, v]);
        }
    }

    #[test]
    fn slow_crop_pads_small_images() {
        let img = ImageU8::new(2, 1, 1, vec![7, 9]).unwrap();
        let out = center_crop_slow(&img, 3, 4);
        assert_eq!((out.width, out.height), (4, 3));
        assert_eq!(out.data, vec![0, 0, 0, 0, 0, 7, 9, 0, 0, 0, 0, 0]);
    }
}
