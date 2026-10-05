//! Typed image processors built from `preprocessor_config.json`.

use crate::config::{PreprocessorConfig, SizeSpec, resize_output_image_size};
use crate::error::{Error, Result};
use crate::image::ops::{self, FloatSemantics, Normalize};
use crate::image::{ImageU8, PassOrder, Resample, TorchInterp, pil_resize, torch_resize};
use ndarray::{Array3, Array4, Axis};

pub const OPENAI_CLIP_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
pub const OPENAI_CLIP_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];
pub const IMAGENET_STANDARD_MEAN: [f64; 3] = [0.5, 0.5, 0.5];
pub const IMAGENET_STANDARD_STD: [f64; 3] = [0.5, 0.5, 0.5];
pub const IMAGENET_DEFAULT_MEAN: [f64; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_DEFAULT_STD: [f64; 3] = [0.229, 0.224, 0.225];

/// Which `transformers` implementation to reproduce.
///
/// `transformers` v5 ships two classes per processor: `XImageProcessor`
/// (torchvision, the default `AutoImageProcessor` picks when torchvision is installed) and
/// `XImageProcessorPil` (PIL + numpy, the v4 "slow" processor). They differ by +-1 in
/// resized pixel values, so pick the one your model's Python reference used.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Backend {
    /// `TorchvisionBackend` (`backend="torchvision"`, v4 `use_fast=True`).
    #[default]
    Torchvision,
    /// `PilBackend` (`backend="pil"`, v4 `use_fast=False` / slow processors).
    Pil,
}

/// How `do_convert_rgb` turns non-RGB inputs into RGB.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RgbConversion {
    /// transformers v5: `PIL.Image.convert("RGB")` (alpha is dropped).
    #[default]
    Convert,
    /// transformers 4.4x-4.5x: alpha-composite on white, then convert.
    CompositeOnWhite,
}

/// The processor families implemented here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessorKind {
    Clip,
    Vit,
    Siglip,
    ConvNext,
    Blip,
}

impl ProcessorKind {
    pub fn from_type_name(name: &str) -> Option<ProcessorKind> {
        Some(match name {
            "CLIPImageProcessor" => ProcessorKind::Clip,
            "ViTImageProcessor" => ProcessorKind::Vit,
            "SiglipImageProcessor" => ProcessorKind::Siglip,
            "ConvNextImageProcessor" => ProcessorKind::ConvNext,
            "BlipImageProcessor" => ProcessorKind::Blip,
            _ => return None,
        })
    }

    /// `transformers`' `IMAGE_PROCESSOR_MAPPING_NAMES` for the model types covered here
    /// (used when a config lacks `image_processor_type`).
    pub fn from_model_type(model_type: &str) -> Option<ProcessorKind> {
        Some(match model_type {
            "clip" | "clip_vision_model" => ProcessorKind::Clip,
            "vit" | "vit_mae" | "vit_msn" => ProcessorKind::Vit,
            "siglip" | "siglip_vision_model" => ProcessorKind::Siglip,
            "convnext" | "convnextv2" => ProcessorKind::ConvNext,
            "blip" => ProcessorKind::Blip,
            _ => return None,
        })
    }

    pub fn class_name(self) -> &'static str {
        match self {
            ProcessorKind::Clip => "CLIPImageProcessor",
            ProcessorKind::Vit => "ViTImageProcessor",
            ProcessorKind::Siglip => "SiglipImageProcessor",
            ProcessorKind::ConvNext => "ConvNextImageProcessor",
            ProcessorKind::Blip => "BlipImageProcessor",
        }
    }
}

/// Class-level defaults (the attributes on e.g. `CLIPImageProcessor` in transformers v5).
struct Defaults {
    resample: Resample,
    mean: [f64; 3],
    std: [f64; 3],
    size: SizeSpec,
    default_to_square: bool,
    crop_size: Option<(usize, usize)>,
    do_resize: bool,
    do_center_crop: bool,
    do_rescale: bool,
    do_normalize: bool,
    do_convert_rgb: bool,
}

fn defaults(kind: ProcessorKind) -> Defaults {
    let base = Defaults {
        resample: Resample::Bicubic,
        mean: IMAGENET_STANDARD_MEAN,
        std: IMAGENET_STANDARD_STD,
        size: SizeSpec::HeightWidth { height: 224, width: 224 },
        default_to_square: true,
        crop_size: None,
        do_resize: true,
        do_center_crop: false,
        do_rescale: true,
        do_normalize: true,
        do_convert_rgb: false,
    };
    match kind {
        ProcessorKind::Clip => Defaults {
            mean: OPENAI_CLIP_MEAN,
            std: OPENAI_CLIP_STD,
            size: SizeSpec::ShortestEdge(224),
            default_to_square: false,
            crop_size: Some((224, 224)),
            do_center_crop: true,
            do_convert_rgb: true,
            ..base
        },
        ProcessorKind::Vit => Defaults { resample: Resample::Bilinear, ..base },
        ProcessorKind::Siglip => Defaults { default_to_square: false, do_convert_rgb: true, ..base },
        ProcessorKind::ConvNext => Defaults { size: SizeSpec::ShortestEdge(384), default_to_square: false, ..base },
        ProcessorKind::Blip => Defaults {
            mean: OPENAI_CLIP_MEAN,
            std: OPENAI_CLIP_STD,
            size: SizeSpec::HeightWidth { height: 384, width: 384 },
            do_convert_rgb: true,
            ..base
        },
    }
}

/// A ready-to-run image processor. All fields are public so they can be tweaked after
/// loading (e.g. `proc.backend = Backend::Pil`).
#[derive(Clone, Debug)]
pub struct ImageProcessor {
    pub kind: ProcessorKind,
    pub backend: Backend,
    pub do_convert_rgb: bool,
    pub rgb_conversion: RgbConversion,
    pub do_resize: bool,
    pub size: SizeSpec,
    pub resample: Resample,
    pub do_center_crop: bool,
    pub crop_size: Option<(usize, usize)>,
    pub do_rescale: bool,
    pub rescale_factor: f64,
    pub do_normalize: bool,
    pub image_mean: Vec<f64>,
    pub image_std: Vec<f64>,
    /// ConvNeXt only.
    pub crop_pct: f64,
    /// Pillow pass ordering for the PIL backend (Pillow <= 12.3 by default).
    pub pil_pass_order: PassOrder,
}

impl ImageProcessor {
    /// Build a processor from a parsed config. The processor class is taken from
    /// `image_processor_type` / `feature_extractor_type`.
    pub fn from_config(cfg: &PreprocessorConfig) -> Result<Self> {
        let name = cfg.processor_type().ok_or_else(|| Error::Config("missing image_processor_type".into()))?;
        let kind = ProcessorKind::from_type_name(&name).ok_or(Error::UnsupportedProcessor(name))?;
        Self::from_config_as(cfg, kind)
    }

    /// Build a processor of an explicit kind (ignores the type name in the config).
    pub fn from_config_as(cfg: &PreprocessorConfig, kind: ProcessorKind) -> Result<Self> {
        let d = defaults(kind);
        let default_to_square = cfg.default_to_square.unwrap_or(d.default_to_square);
        let size = match &cfg.size {
            Some(v) => SizeSpec::from_value(v, default_to_square)?,
            None => d.size,
        };
        let crop_size = match &cfg.crop_size {
            // crop_size always uses default_to_square=True.
            Some(v) => Some(
                SizeSpec::from_value(v, true)?
                    .height_width()
                    .ok_or_else(|| Error::Config("crop_size must have 'height' and 'width'".into()))?,
            ),
            None => d.crop_size,
        };
        let resample = match cfg.resample {
            Some(r) => Resample::from_pil(r).ok_or_else(|| Error::Config(format!("unknown resample {r}")))?,
            None => d.resample,
        };
        let do_center_crop = cfg.do_center_crop.unwrap_or(d.do_center_crop);
        if do_center_crop && crop_size.is_none() {
            return Err(Error::Config("do_center_crop is set but crop_size is missing".into()));
        }
        Ok(ImageProcessor {
            kind,
            backend: Backend::default(),
            do_convert_rgb: cfg.do_convert_rgb.unwrap_or(d.do_convert_rgb),
            rgb_conversion: RgbConversion::default(),
            do_resize: cfg.do_resize.unwrap_or(d.do_resize),
            size,
            resample,
            do_center_crop,
            crop_size,
            do_rescale: cfg.do_rescale.unwrap_or(d.do_rescale),
            rescale_factor: cfg.rescale_factor.unwrap_or(1.0 / 255.0),
            do_normalize: cfg.do_normalize.unwrap_or(d.do_normalize),
            image_mean: cfg.image_mean.as_ref().map(|m| m.to_vec()).unwrap_or(d.mean.to_vec()),
            image_std: cfg.image_std.as_ref().map(|m| m.to_vec()).unwrap_or(d.std.to_vec()),
            crop_pct: cfg.crop_pct.unwrap_or(224.0 / 256.0),
            pil_pass_order: PassOrder::default(),
        })
    }

    pub fn from_json_str(s: &str) -> Result<Self> {
        Self::from_config(&PreprocessorConfig::from_json_str(s)?)
    }

    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    fn resize_to(&self, img: &ImageU8, h: usize, w: usize) -> Result<ImageU8> {
        resize_with_backend(img, h, w, self.backend, self.resample, self.pil_pass_order)
    }

    fn center_crop(&self, img: &ImageU8, h: usize, w: usize) -> Result<ImageU8> {
        if h == 0 || w == 0 {
            return Err(Error::Config(format!("crop size {h}x{w} is empty")));
        }
        // Crops larger than the image are zero-padded: bound the canvas.
        crate::limits::check_alloc("center crop canvas", &[h.max(img.height), w.max(img.width)])?;
        Ok(match self.backend {
            Backend::Pil => ops::center_crop_slow(img, h, w),
            Backend::Torchvision => ops::center_crop_torchvision(img, h, w),
        })
    }

    fn resize(&self, img: &ImageU8) -> Result<ImageU8> {
        if self.kind == ProcessorKind::ConvNext {
            let SizeSpec::ShortestEdge(s) = self.size else {
                return Err(Error::Config("ConvNext size must contain 'shortest_edge'".into()));
            };
            return if s < 384 {
                let resize_shortest = (s as f64 / self.crop_pct) as usize;
                let (h, w) = resize_output_image_size(img.height, img.width, resize_shortest);
                let r = self.resize_to(img, h, w)?;
                self.center_crop(&r, s, s)
            } else {
                self.resize_to(img, s, s)
            };
        }
        let (h, w) = self.size.output_size(img.height, img.width)?;
        self.resize_to(img, h, w)
    }

    /// Run every step up to (not including) rescale/normalize and return the `u8` image
    /// (HWC) that would be normalized. Useful for debugging mismatches.
    pub fn preprocess_u8(&self, img: &ImageU8) -> Result<ImageU8> {
        // Borrow the input until a step produces a new image (no copy of large RGB inputs).
        let converted;
        let mut cur: std::borrow::Cow<'_, ImageU8> = if self.do_convert_rgb && img.channels != 3 {
            converted = convert_rgb(img, self.rgb_conversion);
            std::borrow::Cow::Borrowed(&converted)
        } else {
            std::borrow::Cow::Borrowed(img)
        };
        if self.do_resize {
            cur = std::borrow::Cow::Owned(self.resize(&cur)?);
        }
        if self.do_center_crop {
            let (h, w) = self.crop_size.expect("validated");
            cur = std::borrow::Cow::Owned(self.center_crop(&cur, h, w)?);
        }
        Ok(cur.into_owned())
    }

    /// Preprocess one image into a CHW `f32` array (`pixel_values[i]`).
    pub fn preprocess(&self, img: &ImageU8) -> Result<Array3<f32>> {
        let cur = self.preprocess_u8(img)?;
        self.normalize(&cur)
    }

    fn normalize(&self, img: &ImageU8) -> Result<Array3<f32>> {
        let luts = normalize_luts(
            img.channels,
            self.backend,
            self.do_rescale,
            self.rescale_factor,
            self.do_normalize,
            &self.image_mean,
            &self.image_std,
        )?;
        Ok(ops::apply_luts_chw(img, &luts))
    }

    /// Preprocess a batch into `(N, C, H, W)`; all outputs must share a shape (as with
    /// `return_tensors="pt"`/`"np"` in Python).
    pub fn preprocess_batch(&self, images: &[ImageU8]) -> Result<Array4<f32>> {
        let outs = self.preprocess_many(images)?;
        stack(&outs)
    }

    /// Preprocess several images without requiring a common output shape.
    pub fn preprocess_many(&self, images: &[ImageU8]) -> Result<Vec<Array3<f32>>> {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            images.par_iter().map(|im| self.preprocess(im)).collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            images.iter().map(|im| self.preprocess(im)).collect()
        }
    }

    /// Decode a file and preprocess it.
    #[cfg(feature = "decode")]
    pub fn preprocess_path(&self, path: impl AsRef<std::path::Path>) -> Result<Array3<f32>> {
        self.preprocess(&crate::image::load_image(path)?)
    }
}

/// Resize `img` to `h x w` with the given backend's resampler.
pub(crate) fn resize_with_backend(
    img: &ImageU8,
    h: usize,
    w: usize,
    backend: Backend,
    resample: Resample,
    pass_order: PassOrder,
) -> Result<ImageU8> {
    if h == 0 || w == 0 {
        return Err(Error::Image(format!("resize target {h}x{w} is empty")));
    }
    crate::limits::check_alloc("resize target", &[h, w])?;
    match backend {
        Backend::Pil => {
            if img.channels != 1 && img.channels != 3 {
                return Err(Error::Image(format!(
                    "PIL backend resize of {}-channel images (premultiplied alpha) is not supported; \
                     enable do_convert_rgb",
                    img.channels
                )));
            }
            Ok(pil_resize::resize(img, w, h, resample, pass_order))
        }
        Backend::Torchvision => {
            let mode = TorchInterp::from_pil(resample)?;
            Ok(torch_resize::resize(img, w, h, mode))
        }
    }
}

/// Per-channel rescale+normalize lookup tables (`f(0..=255)`) with the float semantics of
/// `backend`, or the error `transformers` raises when mean/std do not match the channels.
pub(crate) fn normalize_luts(
    c: usize,
    backend: Backend,
    do_rescale: bool,
    rescale_factor: f64,
    do_normalize: bool,
    mean: &[f64],
    std: &[f64],
) -> Result<Vec<[f32; 256]>> {
    if do_normalize {
        for (name, v) in [("image_mean", mean), ("image_std", std)] {
            if v.is_empty() {
                return Err(Error::Config(format!("{name} is empty")));
            }
            if v.len() != 1 && v.len() != mean.len().max(std.len()) {
                return Err(Error::Config(format!(
                    "image_mean and image_std have different lengths ({} and {})",
                    mean.len(),
                    std.len()
                )));
            }
        }
    }
    let n_mean = mean.len().max(std.len());
    if !do_normalize || n_mean == 1 || n_mean == c {
        let norm = Normalize { do_rescale, rescale_factor, do_normalize, mean: mean.to_vec(), std: std.to_vec() };
        let sem = match backend {
            Backend::Pil => FloatSemantics::Numpy,
            Backend::Torchvision => FloatSemantics::TorchFused,
        };
        return Ok(norm.luts(c, sem));
    }
    match backend {
        // numpy normalize: "mean must have {num_channels} elements".
        Backend::Pil => Err(Error::Image(format!(
            "mean must have {c} elements if it is an iterable, got {n_mean} (enable do_convert_rgb)"
        ))),
        // In-place `sub_` cannot broadcast (1, H, W) to (3, H, W); 2/4 channels already
        // fail channel-format inference in transformers.
        Backend::Torchvision => Err(Error::Image(format!(
            "cannot broadcast {c} channels against {n_mean} mean values (enable do_convert_rgb)"
        ))),
    }
}

/// Apply `do_convert_rgb` with the chosen conversion.
pub(crate) fn convert_rgb(img: &ImageU8, conv: RgbConversion) -> ImageU8 {
    match conv {
        RgbConversion::Convert => ops::convert_to_rgb(img),
        RgbConversion::CompositeOnWhite => ops::convert_to_rgb_composite(img),
    }
}

/// Stack CHW arrays into NCHW.
pub fn stack(arrays: &[Array3<f32>]) -> Result<Array4<f32>> {
    if arrays.is_empty() {
        return Err(Error::Image("empty batch".into()));
    }
    let shape = arrays[0].dim();
    if arrays.iter().any(|a| a.dim() != shape) {
        return Err(Error::Image("images in a batch have different output shapes".into()));
    }
    let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
    ndarray::stack(Axis(0), &views).map_err(|e| Error::Image(e.to_string()))
}

#[cfg(feature = "candle")]
pub fn to_candle(arr: &Array4<f32>, device: &candle_core::Device) -> Result<candle_core::Tensor> {
    let (n, c, h, w) = arr.dim();
    let data: Vec<f32> = arr.iter().copied().collect();
    candle_core::Tensor::from_vec(data, (n, c, h, w), device).map_err(|e| Error::Image(e.to_string()))
}
