//! # hf-processors
//!
//! Hugging Face `transformers` preprocessors reproduced in Rust from
//! `preprocessor_config.json`, golden-tested against Python.
//!
//! ```no_run
//! use hf_processors::{AutoImageProcessor, Backend};
//!
//! let proc = AutoImageProcessor::from_pretrained("openai/clip-vit-base-patch32")?;
//! let pixel_values = proc.preprocess_path("cat.png")?; // (3, 224, 224) f32
//! // Match the PIL ("slow") implementation instead of torchvision:
//! let slow = proc.clone().with_backend(Backend::Pil);
//! # Ok::<(), hf_processors::Error>(())
//! ```
//!
//! Supported: `CLIPImageProcessor`, `ViTImageProcessor`, `SiglipImageProcessor`,
//! `ConvNextImageProcessor`, `BlipImageProcessor`, `Qwen2VLImageProcessor` (both the
//! torchvision and the PIL backends of transformers v5) and `WhisperFeatureExtractor`.

pub mod audio;
pub mod config;
pub mod error;
pub mod hub;
pub mod image;
pub mod limits;
pub mod processor;
pub mod qwen2_vl;

pub use audio::whisper::{Padding, WhisperFeatureExtractor, WhisperFeatures, WhisperOptions};
pub use config::{PreprocessorConfig, SizeSpec};
pub use error::{Error, Result};
pub use image::{ImageU8, PassOrder, Resample};
#[cfg(feature = "decode")]
pub use image::{decode_image, load_image};
pub use limits::{DEFAULT_MAX_IMAGE_PIXELS, max_image_pixels, set_max_image_pixels};
pub use processor::{Backend, ImageProcessor, ProcessorKind, RgbConversion};
pub use qwen2_vl::{Qwen2VLImageProcessor, Qwen2VLOutput, smart_resize};

/// `transformers.AutoImageProcessor` equivalent for the fixed-size processors
/// (CLIP, ViT, SigLIP, ConvNeXt, BLIP). Use [`AutoProcessor`] to also get Qwen2-VL and
/// Whisper.
pub struct AutoImageProcessor;

impl AutoImageProcessor {
    /// Load from a Hub repo id (`org/name[@revision]`), a directory containing
    /// `preprocessor_config.json`, or the path of the JSON file itself.
    ///
    /// When the config has no `image_processor_type` (e.g. `google/vit-base-patch16-224`),
    /// the processor class is inferred from `model_type` in the repo's `config.json`, like
    /// `transformers` does.
    pub fn from_pretrained(repo_or_path: &str) -> Result<ImageProcessor> {
        match AutoProcessor::from_pretrained(repo_or_path)? {
            Processor::Image(p) => Ok(p),
            other => Err(Error::UnsupportedProcessor(format!(
                "{} is not a fixed-size image processor; use AutoProcessor",
                other.class_name()
            ))),
        }
    }

    pub fn from_config(cfg: &PreprocessorConfig) -> Result<ImageProcessor> {
        ImageProcessor::from_config(cfg)
    }
}

/// Any supported processor.
#[derive(Clone, Debug)]
pub enum Processor {
    Image(ImageProcessor),
    Qwen2VL(Qwen2VLImageProcessor),
    Whisper(WhisperFeatureExtractor),
}

impl Processor {
    /// The `transformers` class this reproduces (torchvision-backend name for images).
    pub fn class_name(&self) -> &'static str {
        match self {
            Processor::Image(p) => p.kind.class_name(),
            Processor::Qwen2VL(_) => "Qwen2VLImageProcessor",
            Processor::Whisper(_) => "WhisperFeatureExtractor",
        }
    }
}

/// `transformers.AutoProcessor`-like loader covering image processors and audio feature
/// extractors (it does not load tokenizers).
pub struct AutoProcessor;

/// `IMAGE_PROCESSOR_MAPPING_NAMES` / `FEATURE_EXTRACTOR_MAPPING_NAMES` for the model types
/// covered here (used when a config has no processor type).
pub fn processor_type_for_model_type(model_type: &str) -> Option<&'static str> {
    if let Some(kind) = ProcessorKind::from_model_type(model_type) {
        return Some(kind.class_name());
    }
    Some(match model_type {
        "qwen2_vl" | "qwen2_5_vl" | "qwen2_5_omni" | "qwen3_vl" | "qwen3_vl_moe" | "qwen3_5" | "qwen3_5_moe"
        | "qwen3_omni_moe" | "colqwen2" => "Qwen2VLImageProcessor",
        "whisper" => "WhisperFeatureExtractor",
        _ => return None,
    })
}

impl AutoProcessor {
    /// Load from a Hub repo id, a directory or a `preprocessor_config.json` path (see
    /// [`AutoImageProcessor::from_pretrained`]).
    pub fn from_pretrained(repo_or_path: &str) -> Result<Processor> {
        let path = hub::resolve_config(repo_or_path)?;
        let cfg = PreprocessorConfig::from_file(path)?;
        if cfg.processor_type().is_some() {
            return Self::from_config(&cfg);
        }
        let model_cfg = hub::resolve_file(repo_or_path, "config.json")?;
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(model_cfg)?)?;
        let model_type = v.get("model_type").and_then(|m| m.as_str()).unwrap_or_default();
        let name = processor_type_for_model_type(model_type).ok_or_else(|| {
            Error::UnsupportedProcessor(format!("no image_processor_type and model_type `{model_type}`"))
        })?;
        Self::from_config_as(&cfg, name)
    }

    pub fn from_config(cfg: &PreprocessorConfig) -> Result<Processor> {
        let name = cfg
            .processor_type()
            .ok_or_else(|| Error::Config("missing image_processor_type / feature_extractor_type".into()))?;
        Self::from_config_as(cfg, &name)
    }

    /// Build the processor class `name` (aliases like `...Fast`/`...Pil` already stripped).
    pub fn from_config_as(cfg: &PreprocessorConfig, name: &str) -> Result<Processor> {
        Ok(match name {
            "WhisperFeatureExtractor" => Processor::Whisper(WhisperFeatureExtractor::from_config(cfg)?),
            "Qwen2VLImageProcessor" | "Qwen2_5_VLImageProcessor" => {
                Processor::Qwen2VL(Qwen2VLImageProcessor::from_config(cfg)?)
            }
            other => {
                let kind = ProcessorKind::from_type_name(other)
                    .ok_or_else(|| Error::UnsupportedProcessor(other.to_string()))?;
                Processor::Image(ImageProcessor::from_config_as(cfg, kind)?)
            }
        })
    }
}
