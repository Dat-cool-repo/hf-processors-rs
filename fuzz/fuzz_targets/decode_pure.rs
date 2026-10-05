//! Decode arbitrary bytes with the pure-Rust decoders (`image` crate PNG, `jpeg-decoder`),
//! then preprocess the result with one of the golden processors.
#![no_main]
use libfuzzer_sys::fuzz_target;

#[path = "common.rs"]
mod common;

fuzz_target!(|data: &[u8]| {
    common::init();
    if let Ok(img) = hf_processors::image::decode_image_pure_rust(data) {
        common::run_image(&img, data.last().copied().unwrap_or(0));
    }
});
