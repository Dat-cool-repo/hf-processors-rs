//! Shared helpers for the fuzz targets.
#![allow(dead_code)]

use hf_processors::{AutoProcessor, Backend, ImageU8, PreprocessorConfig, Processor};
use std::sync::{Once, OnceLock};

/// Process-wide settings for every target: a small pixel limit keeps each run well under
/// `-rss_limit_mb=2048` (the limit checks themselves are under test), no warnings on stderr.
pub fn init() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        hf_processors::set_max_image_pixels(Some(1 << 20));
        hf_processors::limits::set_warning_handler(None);
    });
}

pub const CONFIGS: &[&str] = &[
    include_str!("../../golden/configs/openai_clip-vit-base-patch32.json"),
    include_str!("../../golden/configs/google_siglip-so400m-patch14-384.json"),
    include_str!("../../golden/configs/google_vit-base-patch16-224.json"),
    include_str!("../../golden/configs/facebook_convnext-tiny-224.json"),
    include_str!("../../golden/configs/facebook_convnext-base-384.json"),
    include_str!("../../golden/configs/Salesforce_blip-image-captioning-base.json"),
    include_str!("../../golden/configs/Qwen_Qwen2-VL-2B-Instruct.json"),
    include_str!("../../golden/configs/Qwen_Qwen3-VL-2B-Instruct.json"),
];

/// Every golden image config, with both backends.
pub fn processors() -> &'static [Processor] {
    static P: OnceLock<Vec<Processor>> = OnceLock::new();
    P.get_or_init(|| {
        let mut out = Vec::new();
        for (i, json) in CONFIGS.iter().enumerate() {
            let cfg = PreprocessorConfig::from_json_str(json).unwrap();
            // ViT's config has no image_processor_type (it is inferred from config.json).
            let p = if i == 2 {
                AutoProcessor::from_config_as(&cfg, "ViTImageProcessor").unwrap()
            } else {
                AutoProcessor::from_config(&cfg).unwrap()
            };
            for b in [Backend::Torchvision, Backend::Pil] {
                out.push(with_backend(p.clone(), b));
            }
        }
        out
    })
}

pub fn with_backend(mut p: Processor, b: Backend) -> Processor {
    match &mut p {
        Processor::Image(p) => p.backend = b,
        Processor::Qwen2VL(p) => p.backend = b,
        Processor::Whisper(_) => {}
    }
    p
}

/// Run `p` on `img`; errors are fine, panics are not. Checks the output shapes.
pub fn run_processor(p: &Processor, img: &ImageU8) {
    match p {
        Processor::Image(p) => {
            if let Ok(out) = p.preprocess(img) {
                let (c, h, w) = out.dim();
                let want_c = if p.do_convert_rgb { 3 } else { img.channels };
                assert_eq!(c, want_c, "channels");
                if p.do_center_crop {
                    assert_eq!((h, w), p.crop_size.unwrap(), "crop size");
                }
            }
        }
        Processor::Qwen2VL(q) => {
            if let Ok(out) = q.preprocess_batch(std::slice::from_ref(img)) {
                let g = out.image_grid_thw.row(0).to_vec();
                assert_eq!(out.pixel_values.nrows() as i64, g[0] * g[1] * g[2], "patch rows");
                let c = if q.do_convert_rgb { 3 } else { img.channels };
                assert_eq!(out.pixel_values.ncols(), q.patch_dim(c), "patch dim");
                assert_eq!(g[1] % q.merge_size as i64, 0);
                assert_eq!(g[2] % q.merge_size as i64, 0);
            }
        }
        Processor::Whisper(_) => {}
    }
}

/// Run one of the golden processors, picked by `sel`.
pub fn run_image(img: &ImageU8, sel: u8) {
    assert_eq!(img.data.len(), img.width * img.height * img.channels, "decoded buffer size");
    assert!((1..=4).contains(&img.channels));
    let procs = processors();
    run_processor(&procs[sel as usize % procs.len()], img);
}

/// A deterministic test image (pseudo-random pixels from `seed`, or a constant).
pub fn synth_image(w: usize, h: usize, c: usize, seed: u64) -> ImageU8 {
    let mut s = seed | 1;
    let data = (0..w * h * c)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 24) as u8
        })
        .collect();
    ImageU8::new(w, h, c, data).unwrap()
}
