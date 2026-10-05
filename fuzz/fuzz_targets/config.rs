//! `preprocessor_config.json` parsing and processor construction from arbitrary JSON text
//! (inputs starting with `{`), or from arbitrary field values set directly, including NaN /
//! infinite / zero / negative mean, std, rescale factor and crop_pct, absurd sizes and
//! missing fields. Every processor that builds is run on a few small images (both backends).
#![no_main]
use arbitrary::{Arbitrary, Unstructured};
use hf_processors::config::{FloatOrVec, SizeDictRaw, SizeValue};
use hf_processors::{AutoProcessor, Backend, PreprocessorConfig, Processor, WhisperOptions};
use libfuzzer_sys::fuzz_target;

#[path = "common.rs"]
mod common;

const CLASSES: &[&str] = &[
    "CLIPImageProcessor",
    "ViTImageProcessor",
    "SiglipImageProcessor",
    "ConvNextImageProcessor",
    "BlipImageProcessor",
    "Qwen2VLImageProcessor",
    "WhisperFeatureExtractor",
];

#[derive(Arbitrary, Debug)]
enum Size {
    Int(u32),
    List(Vec<u32>),
    Dict { h: Option<u32>, w: Option<u32>, se: Option<u32>, le: Option<u32>, mh: Option<u32>, mw: Option<u32> },
}

impl Size {
    fn value(self) -> SizeValue {
        match self {
            Size::Int(v) => SizeValue::Int(v),
            Size::List(v) => SizeValue::List(v),
            Size::Dict { h, w, se, le, mh, mw } => SizeValue::Dict(SizeDictRaw {
                height: h,
                width: w,
                shortest_edge: se,
                longest_edge: le,
                max_height: mh,
                max_width: mw,
            }),
        }
    }
}

#[derive(Arbitrary, Debug)]
enum Floats {
    Scalar(f64),
    Vec(Vec<f64>),
}

#[derive(Arbitrary, Debug)]
struct Fields {
    class: u8,
    do_convert_rgb: Option<bool>,
    do_resize: Option<bool>,
    size: Option<Size>,
    default_to_square: Option<bool>,
    resample: Option<i64>,
    do_center_crop: Option<bool>,
    crop_size: Option<Size>,
    do_rescale: Option<bool>,
    rescale_factor: Option<f64>,
    do_normalize: Option<bool>,
    image_mean: Option<Floats>,
    image_std: Option<Floats>,
    crop_pct: Option<f64>,
    min_pixels: Option<u64>,
    max_pixels: Option<u64>,
    patch_size: Option<u16>,
    temporal_patch_size: Option<u8>,
    merge_size: Option<u8>,
    feature_size: Option<u16>,
    sampling_rate: Option<u32>,
    hop_length: Option<u16>,
    chunk_length: Option<u8>,
    n_fft: Option<u16>,
    padding_value: Option<f64>,
    dither: Option<f64>,
}

fn floats(f: Floats) -> FloatOrVec {
    match f {
        Floats::Scalar(v) => FloatOrVec::Scalar(v),
        Floats::Vec(v) => FloatOrVec::Vec(v),
    }
}

fn from_fields(f: Fields) -> PreprocessorConfig {
    PreprocessorConfig {
        image_processor_type: Some(CLASSES[f.class as usize % CLASSES.len()].to_string()),
        do_convert_rgb: f.do_convert_rgb,
        do_resize: f.do_resize,
        size: f.size.map(Size::value),
        default_to_square: f.default_to_square,
        resample: f.resample,
        do_center_crop: f.do_center_crop,
        crop_size: f.crop_size.map(Size::value),
        do_rescale: f.do_rescale,
        rescale_factor: f.rescale_factor,
        do_normalize: f.do_normalize,
        image_mean: f.image_mean.map(floats),
        image_std: f.image_std.map(floats),
        crop_pct: f.crop_pct,
        min_pixels: f.min_pixels.map(|v| v as usize),
        max_pixels: f.max_pixels.map(|v| v as usize),
        patch_size: f.patch_size.map(usize::from),
        temporal_patch_size: f.temporal_patch_size.map(usize::from),
        merge_size: f.merge_size.map(usize::from),
        feature_size: f.feature_size.map(usize::from),
        sampling_rate: f.sampling_rate,
        hop_length: f.hop_length.map(usize::from),
        chunk_length: f.chunk_length.map(usize::from),
        n_fft: f.n_fft.map(usize::from),
        padding_value: f.padding_value,
        dither: f.dither,
        ..Default::default()
    }
}

fn exercise(p: Processor) {
    match &p {
        Processor::Whisper(fe) => {
            // Bounded by the harness: the mel projection is dense (frames * bins * mels).
            let clip: Vec<f32> = (0..4000).map(|i| (i as f32 * 0.01).sin()).collect();
            let n_fft = fe.n_fft as u128;
            let per_frame = n_fft * (128 - n_fft.leading_zeros() as u128) * 4 + (n_fft / 2 + 1) * fe.feature_size as u128;
            let work = (4000 / fe.hop_length as u128 + 1) * per_frame;
            if work <= 100_000_000 {
                let opts = WhisperOptions { padding: hf_processors::Padding::Longest, ..Default::default() };
                let _ = fe.call(&[&clip], &opts);
            }
        }
        _ => {
            for (w, h, c, seed) in [(13, 7, 3, 1), (1, 1, 1, 2), (3, 40, 4, 3), (64, 48, 2, 4)] {
                let img = common::synth_image(w, h, c, seed);
                for b in [Backend::Torchvision, Backend::Pil] {
                    common::run_processor(&common::with_backend(p.clone(), b), &img);
                }
            }
        }
    }
}

fn try_all(cfg: &PreprocessorConfig) {
    if let Ok(p) = AutoProcessor::from_config(cfg) {
        exercise(p);
    }
    for name in CLASSES {
        if let Ok(p) = AutoProcessor::from_config_as(cfg, name) {
            exercise(p);
        }
    }
}

fuzz_target!(|data: &[u8]| {
    common::init();
    if data.first() == Some(&b'{') {
        if let Ok(s) = std::str::from_utf8(data)
            && let Ok(cfg) = PreprocessorConfig::from_json_str(s)
        {
            let _ = cfg.processor_type();
            try_all(&cfg);
        }
    } else if let Ok(f) = Fields::arbitrary(&mut Unstructured::new(data)) {
        let cfg = from_fields(f);
        // Round-trip through JSON too (NaN / inf serialize as null).
        if let Ok(json) = serde_json::to_string(&cfg) {
            let _ = PreprocessorConfig::from_json_str(&json);
        }
        try_all(&cfg);
    }
});
