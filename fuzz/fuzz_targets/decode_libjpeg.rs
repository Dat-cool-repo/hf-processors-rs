//! Decode arbitrary bytes as JPEG with libjpeg-turbo (mozjpeg-sys, C; instrumented when built
//! with the clang CFLAGS from fuzz/run.sh), then preprocess the result.
#![no_main]
use libfuzzer_sys::fuzz_target;

#[path = "common.rs"]
mod common;

const SOI: [u8; 3] = [0xFF, 0xD8, 0xFF];

fuzz_target!(|data: &[u8]| {
    common::init();
    // Every input goes to the JPEG decoder.
    let buf;
    let bytes = if data.starts_with(&SOI) {
        data
    } else {
        buf = [&SOI[..], data].concat();
        &buf
    };
    if let Ok(img) = hf_processors::decode_image(bytes) {
        common::run_image(&img, data.last().copied().unwrap_or(0));
    }
});
