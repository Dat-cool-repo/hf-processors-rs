//! `Qwen2VLImageProcessor` (transformers v5, both backends): dynamic-resolution resize
//! (`smart_resize`), rescale + normalize, and patching into a flat `pixel_values` of shape
//! `(sum(t * h * w), C * temporal_patch_size * patch_size^2)` plus `image_grid_thw`.
//!
//! The same class (and so this implementation) serves Qwen2-VL, Qwen2.5-VL, Qwen2.5-Omni,
//! Qwen3-VL and Qwen3.5 checkpoints; they differ only in their `preprocessor_config.json`
//! (patch size, min/max pixels, mean/std).
//!
//! Attribution: `smart_resize` and the patch layout are reimplemented from Hugging Face
//! `transformers` (`image_processing_qwen2_vl.py`), Copyright The HuggingFace Team / Qwen team,
//! Apache-2.0 license. See THIRD_PARTY_NOTICES.md.

use crate::config::{PreprocessorConfig, SizeValue, py_round};
use crate::error::{Error, Result};
use crate::image::{ImageU8, PassOrder, Resample};
use crate::processor::{
    Backend, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, RgbConversion, convert_rgb, normalize_luts, resize_with_backend,
};
use ndarray::Array2;

/// Class defaults: `size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}`.
pub const DEFAULT_MIN_PIXELS: usize = 56 * 56;
pub const DEFAULT_MAX_PIXELS: usize = 28 * 28 * 1280;

/// `Qwen2VLImageProcessor` / `Qwen2VLImageProcessorPil`.
#[derive(Clone, Debug)]
pub struct Qwen2VLImageProcessor {
    pub backend: Backend,
    pub do_convert_rgb: bool,
    pub rgb_conversion: RgbConversion,
    pub do_resize: bool,
    /// `size["shortest_edge"]` (or the legacy `min_pixels`): minimum number of pixels.
    pub min_pixels: usize,
    /// `size["longest_edge"]` (or the legacy `max_pixels`): maximum number of pixels.
    pub max_pixels: usize,
    pub resample: Resample,
    pub do_rescale: bool,
    pub rescale_factor: f64,
    pub do_normalize: bool,
    pub image_mean: Vec<f64>,
    pub image_std: Vec<f64>,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub pil_pass_order: PassOrder,
}

/// Output of [`Qwen2VLImageProcessor::preprocess_batch`].
#[derive(Clone, Debug)]
pub struct Qwen2VLOutput {
    /// `(sum over images of grid_t * grid_h * grid_w, C * temporal_patch_size * patch_size^2)`.
    pub pixel_values: Array2<f32>,
    /// `(N, 3)`: `[grid_t, grid_h, grid_w]` per image (`grid_t` is always 1 for images).
    pub image_grid_thw: Array2<i64>,
}

/// `transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize`: the output
/// `(height, width)`, both multiples of `factor`, with `min_pixels <= h * w <= max_pixels`
/// when possible and the aspect ratio kept as closely as possible.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    if height == 0 || width == 0 || factor == 0 {
        return Err(Error::Image(format!("smart_resize of an empty image {width}x{height}")));
    }
    let (hf, wf, ff) = (height as f64, width as f64, factor as f64);
    let ratio = hf.max(wf) / hf.min(wf);
    if ratio > 200.0 {
        return Err(Error::Image(format!("absolute aspect ratio must be smaller than 200, got {ratio}")));
    }
    let mut h_bar = py_round(hf / ff) * factor as i64;
    let mut w_bar = py_round(wf / ff) * factor as i64;
    let fi = factor as i64;
    if (h_bar * w_bar) as u128 > max_pixels as u128 {
        let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
        h_bar = fi.max((hf / beta / ff).floor() as i64 * fi);
        w_bar = fi.max((wf / beta / ff).floor() as i64 * fi);
    } else if ((h_bar * w_bar) as u128) < min_pixels as u128 {
        let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
        h_bar = (hf * beta / ff).ceil() as i64 * fi;
        w_bar = (wf * beta / ff).ceil() as i64 * fi;
    }
    Ok((h_bar as usize, w_bar as usize))
}

impl Qwen2VLImageProcessor {
    /// Build from a parsed config, applying `Qwen2VLImageProcessor.__init__`'s rules:
    /// `size` (dict with `shortest_edge`/`longest_edge`, i.e. min/max pixels) with the
    /// legacy top-level `min_pixels`/`max_pixels` keys taking precedence.
    pub fn from_config(cfg: &PreprocessorConfig) -> Result<Self> {
        let (mut min_pixels, mut max_pixels) = match &cfg.size {
            None => (Some(DEFAULT_MIN_PIXELS), Some(DEFAULT_MAX_PIXELS)),
            Some(SizeValue::Dict(d)) => (d.shortest_edge.map(|v| v as usize), d.longest_edge.map(|v| v as usize)),
            Some(other) => return Err(Error::Config(format!("Qwen2-VL size must be a dict, got {other:?}"))),
        };
        if let Some(m) = cfg.min_pixels {
            min_pixels = Some(m);
        }
        if let Some(m) = cfg.max_pixels {
            max_pixels = Some(m);
        }
        let (Some(min_pixels), Some(max_pixels)) = (min_pixels, max_pixels) else {
            return Err(Error::Config("`size` dict must contain 'shortest_edge' and 'longest_edge' keys".into()));
        };
        let resample = match cfg.resample {
            Some(r) => Resample::from_pil(r).ok_or_else(|| Error::Config(format!("unknown resample {r}")))?,
            None => Resample::Bicubic,
        };
        let p = Qwen2VLImageProcessor {
            backend: Backend::default(),
            do_convert_rgb: cfg.do_convert_rgb.unwrap_or(true),
            rgb_conversion: RgbConversion::default(),
            do_resize: cfg.do_resize.unwrap_or(true),
            min_pixels,
            max_pixels,
            resample,
            do_rescale: cfg.do_rescale.unwrap_or(true),
            rescale_factor: cfg.rescale_factor.unwrap_or(1.0 / 255.0),
            do_normalize: cfg.do_normalize.unwrap_or(true),
            image_mean: cfg.image_mean.as_ref().map(|m| m.to_vec()).unwrap_or(OPENAI_CLIP_MEAN.to_vec()),
            image_std: cfg.image_std.as_ref().map(|m| m.to_vec()).unwrap_or(OPENAI_CLIP_STD.to_vec()),
            patch_size: cfg.patch_size.unwrap_or(14),
            temporal_patch_size: cfg.temporal_patch_size.unwrap_or(2),
            merge_size: cfg.merge_size.unwrap_or(2),
            pil_pass_order: PassOrder::default(),
        };
        if p.patch_size == 0 || p.merge_size == 0 || p.temporal_patch_size == 0 {
            return Err(Error::Config("patch_size, merge_size and temporal_patch_size must be positive".into()));
        }
        Ok(p)
    }

    pub fn from_json_str(s: &str) -> Result<Self> {
        Self::from_config(&PreprocessorConfig::from_json_str(s)?)
    }

    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// `patch_size * merge_size`: both output sides are multiples of it.
    pub fn factor(&self) -> usize {
        self.patch_size * self.merge_size
    }

    /// The resized `(height, width)` for an input of `height x width`.
    pub fn output_size(&self, height: usize, width: usize) -> Result<(usize, usize)> {
        if !self.do_resize {
            return Ok((height, width));
        }
        smart_resize(height, width, self.factor(), self.min_pixels, self.max_pixels)
    }

    /// `get_number_of_image_patches`: rows of `pixel_values` an image of this size produces.
    pub fn number_of_image_patches(&self, height: usize, width: usize) -> Result<usize> {
        let (h, w) = smart_resize(height, width, self.factor(), self.min_pixels, self.max_pixels)?;
        Ok((h / self.patch_size) * (w / self.patch_size))
    }

    /// Row length of `pixel_values` for `channels` input channels.
    pub fn patch_dim(&self, channels: usize) -> usize {
        channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    /// RGB conversion + `smart_resize`: the u8 HWC image that gets normalized and patched.
    pub fn preprocess_u8(&self, img: &ImageU8) -> Result<ImageU8> {
        let converted;
        let cur = if self.do_convert_rgb && img.channels != 3 {
            converted = convert_rgb(img, self.rgb_conversion);
            &converted
        } else {
            img
        };
        if !self.do_resize {
            return Ok(cur.clone());
        }
        let (h, w) = self.output_size(cur.height, cur.width)?;
        if (h, w) == (cur.height, cur.width) {
            return Ok(cur.clone());
        }
        resize_with_backend(cur, h, w, self.backend, self.resample, self.pil_pass_order)
    }

    /// Preprocess one image: `(pixel_values (grid_h * grid_w, patch_dim), [1, grid_h, grid_w])`.
    pub fn preprocess(&self, img: &ImageU8) -> Result<(Array2<f32>, [i64; 3])> {
        let cur = self.preprocess_u8(img)?;
        let luts = normalize_luts(
            cur.channels,
            self.backend,
            self.do_rescale,
            self.rescale_factor,
            self.do_normalize,
            &self.image_mean,
            &self.image_std,
        )?;
        let (gh, gw) = self.grid(&cur)?;
        let dim = self.patch_dim(cur.channels);
        let mut out = vec![0f32; gh * gw * dim];
        self.patchify_into(&cur, &luts, &mut out);
        let arr = Array2::from_shape_vec((gh * gw, dim), out).expect("shape");
        Ok((arr, [1, gh as i64, gw as i64]))
    }

    fn grid(&self, img: &ImageU8) -> Result<(usize, usize)> {
        let f = self.factor();
        if !img.height.is_multiple_of(f) || !img.width.is_multiple_of(f) {
            return Err(Error::Image(format!(
                "image of {}x{} cannot be patched: both sides must be multiples of patch_size * merge_size = {f}",
                img.width, img.height
            )));
        }
        Ok((img.height / self.patch_size, img.width / self.patch_size))
    }

    /// The patch layout of `Qwen2VLImageProcessor.patchify`: rows ordered
    /// `(grid_h / m, grid_w / m, m, m)`, each row `(C, T, P, P)` with the frame repeated `T` times.
    fn patchify_into(&self, img: &ImageU8, luts: &[[f32; 256]], out: &mut [f32]) {
        let (p, m, t) = (self.patch_size, self.merge_size, self.temporal_patch_size);
        let c = img.channels;
        let (gh, gw) = (img.height / p, img.width / p);
        let dim = c * t * p * p;
        let block = p * p;
        let mut row = 0usize;
        for bh in 0..gh / m {
            for bw in 0..gw / m {
                for mh in 0..m {
                    for mw in 0..m {
                        let dst = &mut out[row * dim..(row + 1) * dim];
                        let (y0, x0) = ((bh * m + mh) * p, (bw * m + mw) * p);
                        for (ch, lut) in luts.iter().enumerate() {
                            let base = ch * t * block;
                            {
                                let first = &mut dst[base..base + block];
                                for py in 0..p {
                                    let src = &img.data[((y0 + py) * img.width + x0) * c..];
                                    for px in 0..p {
                                        first[py * p + px] = lut[src[px * c + ch] as usize];
                                    }
                                }
                            }
                            for tt in 1..t {
                                dst.copy_within(base..base + block, base + tt * block);
                            }
                        }
                        row += 1;
                    }
                }
            }
        }
    }

    /// Preprocess several images and concatenate them like `Qwen2VLImageProcessor.__call__`.
    pub fn preprocess_batch(&self, images: &[ImageU8]) -> Result<Qwen2VLOutput> {
        if images.is_empty() {
            return Err(Error::Image("empty batch".into()));
        }
        #[cfg(feature = "rayon")]
        let parts: Vec<(Array2<f32>, [i64; 3])> = {
            use rayon::prelude::*;
            images.par_iter().map(|im| self.preprocess(im)).collect::<Result<_>>()?
        };
        #[cfg(not(feature = "rayon"))]
        let parts: Vec<(Array2<f32>, [i64; 3])> = images.iter().map(|im| self.preprocess(im)).collect::<Result<_>>()?;
        let dim = parts[0].0.ncols();
        if parts.iter().any(|(a, _)| a.ncols() != dim) {
            return Err(Error::Image("images in a batch have different channel counts".into()));
        }
        let rows: usize = parts.iter().map(|(a, _)| a.nrows()).sum();
        let mut data = Vec::with_capacity(rows * dim);
        let mut grid = Vec::with_capacity(parts.len() * 3);
        for (a, g) in &parts {
            data.extend_from_slice(a.as_slice().expect("standard layout"));
            grid.extend_from_slice(g);
        }
        Ok(Qwen2VLOutput {
            pixel_values: Array2::from_shape_vec((rows, dim), data).expect("shape"),
            image_grid_thw: Array2::from_shape_vec((parts.len(), 3), grid).expect("shape"),
        })
    }

    /// Decode a file and preprocess it.
    #[cfg(feature = "decode")]
    pub fn preprocess_path(&self, path: impl AsRef<std::path::Path>) -> Result<Qwen2VLOutput> {
        self.preprocess_batch(&[crate::image::load_image(path)?])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_matches_python() {
        // Values from transformers' smart_resize (factor 28, min 3136, max 12845056 / 200704).
        assert_eq!(smart_resize(480, 640, 28, 3136, 12845056).unwrap(), (476, 644));
        assert_eq!(smart_resize(1, 1, 28, 3136, 12845056).unwrap(), (56, 56));
        assert_eq!(smart_resize(2000, 3000, 28, 3136, 200704).unwrap(), (364, 532));
        assert!(smart_resize(10, 2100, 28, 3136, 12845056).is_err());
    }

    #[test]
    fn patch_layout() {
        // 4x4 single-channel image, patch 1, merge 2, temporal 2 -> rows ordered by 2x2 blocks.
        let p = Qwen2VLImageProcessor {
            backend: Backend::Torchvision,
            do_convert_rgb: false,
            rgb_conversion: RgbConversion::Convert,
            do_resize: false,
            min_pixels: 1,
            max_pixels: 1 << 30,
            resample: Resample::Bicubic,
            do_rescale: false,
            rescale_factor: 1.0,
            do_normalize: false,
            image_mean: vec![0.0],
            image_std: vec![1.0],
            patch_size: 1,
            temporal_patch_size: 2,
            merge_size: 2,
            pil_pass_order: PassOrder::default(),
        };
        let img = ImageU8::new(4, 4, 1, (0..16).collect()).unwrap();
        let (pv, g) = p.preprocess(&img).unwrap();
        assert_eq!(g, [1, 4, 4]);
        let firsts: Vec<f32> = pv.rows().into_iter().map(|r| r[0]).collect();
        assert_eq!(firsts, [0., 1., 4., 5., 2., 3., 6., 7., 8., 9., 12., 13., 10., 11., 14., 15.]);
        assert!(pv.rows().into_iter().all(|r| r[0] == r[1]));
    }
}
