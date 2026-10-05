//! The Pillow and torchvision resize kernels with arbitrary dimensions, channel counts and
//! filters. The SIMD kernels are checked against the portable ones on every input.
#![no_main]
use arbitrary::Arbitrary;
use hf_processors::image::{__set_force_scalar_kernels, PassOrder, Resample, TorchInterp, pil_resize, torch_resize};
use libfuzzer_sys::fuzz_target;

#[path = "common.rs"]
mod common;

#[derive(Arbitrary, Debug)]
struct Input {
    w: u32,
    h: u32,
    channels: u8,
    out_w: u32,
    out_h: u32,
    filter: u8,
    adaptive: bool,
    constant: Option<u8>,
    seed: u64,
}

/// Keep each run fast: at most 2^18 input and 2^20 output pixels (any aspect ratio).
const MAX_IN: u64 = 1 << 18;
const MAX_OUT: u64 = 1 << 20;

fuzz_target!(|inp: Input| {
    common::init();
    let (w, h, ow, oh) = (inp.w as u64, inp.h as u64, inp.out_w as u64, inp.out_h as u64);
    // The horizontal pass runs first: its output is h x ow (the library bounds it the same way).
    if w == 0 || h == 0 || ow == 0 || oh == 0 || w * h > MAX_IN || ow * oh > MAX_OUT || ow * h > MAX_OUT {
        return;
    }
    let (w, h, ow, oh) = (w as usize, h as usize, ow as usize, oh as usize);
    let c = 1 + inp.channels as usize % 4;
    let img = match inp.constant {
        Some(v) => hf_processors::ImageU8::new(w, h, c, vec![v; w * h * c]).unwrap(),
        None => common::synth_image(w, h, c, inp.seed),
    };
    let resample = Resample::from_pil(inp.filter as i64 % 6).unwrap();
    let order = if inp.adaptive { PassOrder::Adaptive } else { PassOrder::HorizontalFirst };

    let run = || {
        let pil = pil_resize::resize(&img, ow, oh, resample, order);
        let tv = TorchInterp::from_pil(resample).ok().map(|m| torch_resize::resize(&img, ow, oh, m));
        (pil, tv)
    };
    let (pil, tv) = run();
    __set_force_scalar_kernels(true);
    let (pil_scalar, tv_scalar) = run();
    __set_force_scalar_kernels(false);

    for out in std::iter::once(&pil).chain(tv.as_ref()) {
        assert_eq!((out.width, out.height, out.channels), (ow, oh, c));
        assert_eq!(out.data.len(), ow * oh * c);
    }
    assert!(pil == pil_scalar, "PIL resize: SIMD and scalar kernels differ");
    assert!(tv == tv_scalar, "torchvision resize: SIMD and scalar kernels differ");
    // A constant image stays constant up to +-1: the fixed-point weights need not sum to exactly
    // 1.0. (Only for moderate downscaling: with thousands of taps the per-tap rounding of
    // Pillow's 22-bit and ATen's int16 weights legitimately adds up to more.)
    if let Some(v) = inp.constant.filter(|_| w <= 16 * ow && h <= 16 * oh) {
        for out in std::iter::once(&pil).chain(tv.as_ref()) {
            assert!(out.data.iter().all(|&x| x.abs_diff(v) <= 1), "constant image changed by more than 1");
        }
    }
});
