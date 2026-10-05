//! `preprocessor_config.json` parsing and the size-dict rules of `transformers`.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// `size` / `crop_size` as found in configs: an int, a `[h, w]` list or a dict.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum SizeValue {
    Int(u32),
    List(Vec<u32>),
    Dict(SizeDictRaw),
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct SizeDictRaw {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shortest_edge: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub longest_edge: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_height: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_width: Option<u32>,
}

/// `image_mean` / `image_std`: a scalar or one value per channel.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum FloatOrVec {
    Scalar(f64),
    Vec(Vec<f64>),
}

impl FloatOrVec {
    pub fn to_vec(&self) -> Vec<f64> {
        match self {
            FloatOrVec::Scalar(v) => vec![*v],
            FloatOrVec::Vec(v) => v.clone(),
        }
    }
}

/// The raw contents of a `preprocessor_config.json`. Every field is optional; missing fields
/// fall back to the processor class defaults, exactly like `transformers`.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct PreprocessorConfig {
    pub image_processor_type: Option<String>,
    pub feature_extractor_type: Option<String>,
    pub processor_class: Option<String>,

    // Image processors.
    pub do_convert_rgb: Option<bool>,
    pub do_resize: Option<bool>,
    pub size: Option<SizeValue>,
    pub default_to_square: Option<bool>,
    pub resample: Option<i64>,
    pub do_center_crop: Option<bool>,
    pub crop_size: Option<SizeValue>,
    pub do_rescale: Option<bool>,
    pub rescale_factor: Option<f64>,
    pub do_normalize: Option<bool>,
    pub image_mean: Option<FloatOrVec>,
    pub image_std: Option<FloatOrVec>,
    pub crop_pct: Option<f64>,

    // Dynamic-resolution VLM processors (Qwen2-VL family). Parsed leniently: other
    // processors use e.g. `patch_size: {"height": .., "width": ..}`, which reads as `None`.
    #[serde(default, deserialize_with = "lenient_usize")]
    pub min_pixels: Option<usize>,
    #[serde(default, deserialize_with = "lenient_usize")]
    pub max_pixels: Option<usize>,
    #[serde(default, deserialize_with = "lenient_usize")]
    pub patch_size: Option<usize>,
    #[serde(default, deserialize_with = "lenient_usize")]
    pub temporal_patch_size: Option<usize>,
    #[serde(default, deserialize_with = "lenient_usize")]
    pub merge_size: Option<usize>,

    // Audio feature extractors (Whisper).
    pub feature_size: Option<usize>,
    pub sampling_rate: Option<u32>,
    pub hop_length: Option<usize>,
    pub chunk_length: Option<usize>,
    pub n_fft: Option<usize>,
    pub padding_value: Option<f64>,
    pub n_samples: Option<usize>,
    pub nb_max_frames: Option<usize>,
    pub return_attention_mask: Option<bool>,
    pub dither: Option<f64>,

    /// Everything else, kept for inspection.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// An optional non-negative integer; any other JSON value deserializes to `None`.
fn lenient_usize<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Option<usize>, D::Error> {
    let v = serde_json::Value::deserialize(d)?;
    Ok(v.as_u64().map(|x| x as usize))
}

impl PreprocessorConfig {
    pub fn from_json_str(s: &str) -> Result<Self> {
        Ok(serde_json::from_str(s)?)
    }

    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let s = std::fs::read_to_string(path.as_ref())?;
        Self::from_json_str(&s)
    }

    /// The processor class name with `transformers` aliases normalized:
    /// `CLIPFeatureExtractor` -> `CLIPImageProcessor`, `...Fast`/`...Pil` suffixes stripped.
    pub fn processor_type(&self) -> Option<String> {
        let raw = self.image_processor_type.as_ref().or(self.feature_extractor_type.as_ref())?;
        let mut name = raw.clone();
        for suffix in ["Fast", "Pil"] {
            if let Some(s) = name.strip_suffix(suffix) {
                name = s.to_string();
            }
        }
        if name.ends_with("FeatureExtractor") && !name.starts_with("Whisper") {
            name = name.replace("FeatureExtractor", "ImageProcessor");
        }
        Some(name)
    }
}

/// A validated size dict (`transformers.image_utils.SizeDict`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SizeSpec {
    HeightWidth { height: usize, width: usize },
    ShortestEdge(usize),
    ShortestLongest { shortest: usize, longest: usize },
    LongestEdge(usize),
    MaxHeightWidth { max_height: usize, max_width: usize },
}

impl SizeSpec {
    /// `transformers.image_processing_utils.get_size_dict`.
    pub fn from_value(v: &SizeValue, default_to_square: bool) -> Result<SizeSpec> {
        let d = match v {
            SizeValue::Int(s) if default_to_square => {
                SizeDictRaw { height: Some(*s), width: Some(*s), ..Default::default() }
            }
            SizeValue::Int(s) => SizeDictRaw { shortest_edge: Some(*s), ..Default::default() },
            SizeValue::List(l) if l.len() == 2 => {
                SizeDictRaw { height: Some(l[0]), width: Some(l[1]), ..Default::default() }
            }
            SizeValue::List(l) => {
                return Err(Error::Config(format!("could not convert size {l:?} to a size dict")));
            }
            SizeValue::Dict(d) => d.clone(),
        };
        let u = |x: Option<u32>| x.map(|v| v as usize);
        Ok(match (u(d.height), u(d.width), u(d.shortest_edge), u(d.longest_edge), u(d.max_height), u(d.max_width)) {
            (Some(height), Some(width), None, None, None, None) => SizeSpec::HeightWidth { height, width },
            (None, None, Some(s), None, None, None) => SizeSpec::ShortestEdge(s),
            (None, None, Some(shortest), Some(longest), None, None) => SizeSpec::ShortestLongest { shortest, longest },
            (None, None, None, Some(l), None, None) => SizeSpec::LongestEdge(l),
            (None, None, None, None, Some(max_height), Some(max_width)) => {
                SizeSpec::MaxHeightWidth { max_height, max_width }
            }
            _ => return Err(Error::Config(format!("invalid size dict {d:?}"))),
        })
    }

    /// Output `(height, width)` of the generic `resize` step for an input of `(h, w)`.
    pub fn output_size(&self, h: usize, w: usize) -> Result<(usize, usize)> {
        Ok(match *self {
            SizeSpec::ShortestLongest { shortest, longest } => size_with_aspect_ratio(h, w, shortest, Some(longest)),
            SizeSpec::ShortestEdge(s) => resize_output_image_size(h, w, s),
            SizeSpec::MaxHeightWidth { max_height, max_width } => {
                let hs = max_height as f64 / h as f64;
                let ws = max_width as f64 / w as f64;
                let m = hs.min(ws);
                ((h as f64 * m) as usize, (w as f64 * m) as usize)
            }
            SizeSpec::HeightWidth { height, width } => (height, width),
            SizeSpec::LongestEdge(_) => {
                return Err(Error::Config(
                    "size must contain 'height' and 'width', 'max_height' and 'max_width', or 'shortest_edge'".into(),
                ));
            }
        })
    }

    pub fn height_width(&self) -> Option<(usize, usize)> {
        match *self {
            SizeSpec::HeightWidth { height, width } => Some((height, width)),
            _ => None,
        }
    }
}

/// `transformers.image_transforms.get_resize_output_image_size(..., default_to_square=False)`.
pub fn resize_output_image_size(h: usize, w: usize, size: usize) -> (usize, usize) {
    let (short, long) = if w <= h { (w, h) } else { (h, w) };
    // Python: int(requested_new_short * long / short) -- exact int product, true division.
    let new_short = size;
    let new_long = ((size as u128 * long as u128) as f64 / short as f64) as usize;
    if w <= h { (new_long, new_short) } else { (new_short, new_long) }
}

/// Python's `round()` on a float (half to even).
pub(crate) fn py_round(x: f64) -> i64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 { (2.0 * (x / 2.0).round()) as i64 } else { r as i64 }
}

/// `transformers.image_transforms.get_size_with_aspect_ratio`.
pub fn size_with_aspect_ratio(h: usize, w: usize, size: usize, max_size: Option<usize>) -> (usize, usize) {
    let (hf, wf) = (h as f64, w as f64);
    let mut size = size as i64;
    let mut raw_size: Option<f64> = None;
    if let Some(max) = max_size {
        let min_o = hf.min(wf);
        let max_o = hf.max(wf);
        if max_o / min_o * size as f64 > max as f64 {
            let raw = max as f64 * min_o / max_o;
            raw_size = Some(raw);
            size = py_round(raw);
        }
    }
    let (h_i, w_i) = (h as i64, w as i64);
    let (oh, ow) = if (h_i <= w_i && h_i == size) || (w_i <= h_i && w_i == size) {
        (h_i, w_i)
    } else if w_i < h_i {
        let oh = match raw_size {
            Some(raw) => (raw * hf / wf) as i64,
            None => ((size * h_i) as f64 / wf) as i64,
        };
        (oh, size)
    } else {
        let ow = match raw_size {
            Some(raw) => (raw * wf / hf) as i64,
            None => ((size * w_i) as f64 / hf) as i64,
        };
        (size, ow)
    };
    (oh as usize, ow as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_dict_conversion() {
        assert_eq!(SizeSpec::from_value(&SizeValue::Int(224), false).unwrap(), SizeSpec::ShortestEdge(224));
        assert_eq!(
            SizeSpec::from_value(&SizeValue::Int(224), true).unwrap(),
            SizeSpec::HeightWidth { height: 224, width: 224 }
        );
    }

    #[test]
    fn shortest_edge_matches_python() {
        // int(224 * 640 / 480) = 298
        assert_eq!(resize_output_image_size(480, 640, 224), (224, 298));
        assert_eq!(resize_output_image_size(640, 480, 224), (298, 224));
        assert_eq!(resize_output_image_size(1, 1, 224), (224, 224));
    }

    #[test]
    fn processor_type_aliases() {
        let c: PreprocessorConfig =
            serde_json::from_str(r#"{"feature_extractor_type": "CLIPFeatureExtractor"}"#).unwrap();
        assert_eq!(c.processor_type().unwrap(), "CLIPImageProcessor");
        let c: PreprocessorConfig =
            serde_json::from_str(r#"{"image_processor_type": "SiglipImageProcessorFast"}"#).unwrap();
        assert_eq!(c.processor_type().unwrap(), "SiglipImageProcessor");
    }
}
