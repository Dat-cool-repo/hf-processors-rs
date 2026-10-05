//! Native reference outputs for `tests/node_check.mjs`, which runs the same cases through the
//! wasm module in Node.js and compares the bytes. Same crate, so the same features as the wasm
//! build (pure-Rust JPEG decoder, no hub / rayon / pil-jpeg).
//!
//! ```text
//! cargo run --release -p hf-processors-wasm --example native_reference -- OUT_DIR GOLDEN_DIR IMAGE...
//! ```
//!
//! Writes `OUT_DIR/<config>__<backend>__<image>.f32` (little-endian float32) per image case and
//! `OUT_DIR/whisper.f32` for a synthetic 2.5 s signal through the Whisper extractor.

use hf_processors::{AutoProcessor, Backend, PreprocessorConfig, Processor, WhisperOptions, decode_image};
use std::path::Path;

/// Keep in sync with tests/node_check.mjs.
const CONFIGS: &[&str] = &[
    "openai_clip-vit-base-patch32.json",
    "google_siglip-so400m-patch14-384.json",
    "facebook_convnext-tiny-224.json",
    "Salesforce_blip-image-captioning-base.json",
    "Qwen_Qwen2-VL-2B-Instruct.json",
];

fn write_f32(path: &Path, v: &[f32]) -> std::io::Result<()> {
    std::fs::write(path, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err("usage: native_reference OUT_DIR GOLDEN_DIR IMAGE...".into());
    }
    let (out, golden) = (Path::new(&args[1]), Path::new(&args[2]));
    std::fs::create_dir_all(out)?;
    for cfg_name in CONFIGS {
        let cfg = PreprocessorConfig::from_file(golden.join("configs").join(cfg_name))?;
        for (backend, bname) in [(Backend::Torchvision, "torchvision"), (Backend::Pil, "pil")] {
            let mut p = AutoProcessor::from_config(&cfg)?;
            match &mut p {
                Processor::Image(p) => p.backend = backend,
                Processor::Qwen2VL(p) => p.backend = backend,
                Processor::Whisper(_) => unreachable!(),
            }
            for image in &args[3..] {
                let img = decode_image(&std::fs::read(image)?)?;
                let data: Vec<f32> = match &p {
                    Processor::Image(p) => p.preprocess(&img)?.into_raw_vec_and_offset().0,
                    Processor::Qwen2VL(p) => {
                        p.preprocess_batch(std::slice::from_ref(&img))?.pixel_values.into_raw_vec_and_offset().0
                    }
                    Processor::Whisper(_) => unreachable!(),
                };
                let name = Path::new(image).file_name().unwrap().to_string_lossy();
                let stem = cfg_name.trim_end_matches(".json");
                write_f32(&out.join(format!("{stem}__{bname}__{name}.f32")), &data)?;
            }
        }
    }
    let cfg = PreprocessorConfig::from_file(golden.join("configs").join("openai_whisper-tiny.json"))?;
    let Processor::Whisper(fe) = AutoProcessor::from_config(&cfg)? else { return Err("not whisper".into()) };
    // Exactly representable (dyadic) samples, so JavaScript builds the identical signal.
    let tone: Vec<f32> = (0..40_000u64).map(|i| ((i * 7919 % 2000) as f32 - 1000.0) / 4096.0).collect();
    let feats = fe.call(&[&tone], &WhisperOptions::default())?.input_features;
    write_f32(&out.join("whisper.f32"), &feats.into_raw_vec_and_offset().0)?;
    println!("wrote reference outputs to {}", out.display());
    Ok(())
}
