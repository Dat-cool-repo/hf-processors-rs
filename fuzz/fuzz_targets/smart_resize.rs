//! Qwen2-VL `smart_resize` with extreme sizes, aspect ratios and pixel budgets, plus the full
//! Qwen2-VL processor (resize + patching) on small synthetic images.
#![no_main]
use arbitrary::Arbitrary;
use hf_processors::{Backend, Qwen2VLImageProcessor, smart_resize};
use libfuzzer_sys::fuzz_target;

#[path = "common.rs"]
mod common;

#[derive(Arbitrary, Debug)]
struct Input {
    h: u64,
    w: u64,
    factor: u64,
    min_pixels: u64,
    max_pixels: u64,
    process: Option<Process>,
}

#[derive(Arbitrary, Debug)]
struct Process {
    patch_size: u8,
    merge_size: u8,
    temporal: u8,
    channels: u8,
    pil: bool,
    filter: u8,
    seed: u64,
}

fuzz_target!(|inp: Input| {
    common::init();
    let (h, w, f) = (inp.h as usize, inp.w as usize, inp.factor as usize);
    let (min, max) = (inp.min_pixels as usize, inp.max_pixels as usize);
    if let Ok((rh, rw)) = smart_resize(h, w, f, min, max) {
        assert!(h > 0 && w > 0 && f > 0);
        assert_eq!(rh % f, 0, "height not a multiple of factor");
        assert_eq!(rw % f, 0, "width not a multiple of factor");
        let ratio = h.max(w) as f64 / h.min(w) as f64;
        assert!(ratio <= 200.0);
    }
    let Some(p) = inp.process else { return };
    if h == 0 || w == 0 || (h as u128) * (w as u128) > 1 << 16 {
        return;
    }
    let json = format!(
        r#"{{"image_processor_type": "Qwen2VLImageProcessor", "min_pixels": {min}, "max_pixels": {max},
            "patch_size": {}, "merge_size": {}, "temporal_patch_size": {}, "resample": {}}}"#,
        p.patch_size,
        p.merge_size,
        p.temporal,
        p.filter % 6
    );
    let Ok(mut q) = Qwen2VLImageProcessor::from_json_str(&json) else { return };
    if p.pil {
        q = q.with_backend(Backend::Pil);
    }
    let img = common::synth_image(w, h, 1 + p.channels as usize % 4, p.seed);
    common::run_processor(&hf_processors::Processor::Qwen2VL(q), &img);
});
