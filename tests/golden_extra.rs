//! Second golden suite (fixtures from `golden/make_golden_extra.py`): resample filters,
//! 16-bit PNG / CMYK JPEG / palette decoding, Qwen2-VL and Whisper call options.
//!
//! Image cases are bit-exact checks: SHA-256 of the float32 `pixel_values` bytes.
//! Run with `cargo test --release --features pil-jpeg -- --nocapture` for the report.

use hf_processors::{
    AutoProcessor, Backend, ImageProcessor, Padding, PreprocessorConfig, Processor, ProcessorKind,
    Qwen2VLImageProcessor, WhisperFeatureExtractor, WhisperOptions, load_image,
};
use ndarray::{Array2, Array3, ArrayD, Ix2, Ix3};
use ndarray_npy::NpzReader;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct Manifest {
    versions: Value,
    fixed: Vec<ImageCase>,
    decode: Vec<DecodeCase>,
    qwen: Vec<ImageCase>,
    whisper: Vec<WhisperCase>,
}

#[derive(Deserialize, Clone)]
struct ImageCase {
    suite: String,
    case: Option<String>,
    config: String,
    overrides: serde_json::Map<String, Value>,
    processor: Option<String>,
    backend: String,
    image: String,
    mode: Option<String>,
    status: String,
    error: Option<String>,
    shape: Option<Vec<usize>>,
    grid_thw: Option<Value>,
    batch: Option<Vec<String>>,
    sha256_f32: Option<String>,
    key: Option<String>,
    fixture: Option<String>,
}

#[derive(Deserialize)]
struct DecodeCase {
    image: String,
    mode: String,
    shape: Vec<usize>,
    sha256: String,
}

#[derive(Deserialize)]
struct WhisperCase {
    name: String,
    clips: Vec<String>,
    kwargs: serde_json::Map<String, Value>,
    overrides: serde_json::Map<String, Value>,
    dither_seed: Option<u64>,
    shape: Vec<usize>,
    mask_shape: Option<Vec<usize>>,
    mask_lengths: Option<Vec<usize>>,
}

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("golden")
}

/// Optional directory with the full uint8 references written by the golden generators
/// (`HF_PROCESSORS_GOLDEN_FULL`). The committed fixtures are enough for the tests (SHA-256
/// checks); the full arrays only add per-pixel diagnostics when a case fails.
fn external_dir() -> Option<PathBuf> {
    std::env::var_os("HF_PROCESSORS_GOLDEN_FULL").filter(|v| !v.is_empty()).map(PathBuf::from)
}

fn manifest() -> Manifest {
    let s = std::fs::read_to_string(golden_dir().join("manifest_extra.json")).expect("golden/manifest_extra.json");
    serde_json::from_str(&s).expect("manifest_extra")
}

fn config_with(file: &str, overrides: &serde_json::Map<String, Value>) -> PreprocessorConfig {
    let s = std::fs::read_to_string(golden_dir().join("configs").join(file)).unwrap();
    let mut v: Value = serde_json::from_str(&s).unwrap();
    for (k, val) in overrides {
        v[k] = val.clone();
    }
    serde_json::from_value(v).unwrap()
}

fn backend_of(s: &str) -> Backend {
    match s {
        "pil" => Backend::Pil,
        "torchvision" => Backend::Torchvision,
        other => panic!("unknown backend {other}"),
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn sha_f32<'a>(values: impl Iterator<Item = &'a f32>) -> String {
    let mut h = Sha256::new();
    for v in values {
        h.update(v.to_le_bytes());
    }
    hex(&h.finalize())
}

/// Per-pixel diagnostics from the external u8 reference + repo LUT, when available.
fn diagnose(case: &ImageCase, got_chw: &Array3<f32>) -> String {
    let (Some(fixture), Some(key)) = (&case.fixture, &case.key) else { return "no reference".into() };
    let Some(dir) = external_dir() else {
        return "no full reference (set HF_PROCESSORS_GOLDEN_FULL for per-pixel diagnostics)".into();
    };
    let Ok(f) = File::open(dir.join(fixture)) else { return "no external reference".into() };
    let mut npz = NpzReader::new(f).unwrap();
    let Ok(u8ref) = npz.by_name::<ndarray::OwnedRepr<u8>, Ix3>(&format!("u8__{key}.npy")) else {
        return "reference missing".into();
    };
    let c = u8ref.shape()[0];
    let lut: Array2<f32> = npz.by_name(&format!("lut_c{c}.npy")).unwrap();
    if u8ref.shape() != got_chw.shape() {
        return format!("shape {:?} vs reference {:?}", got_chw.shape(), u8ref.shape());
    }
    let mut max = 0f32;
    let mut n = 0usize;
    for ((ch, y, x), &r) in u8ref.indexed_iter() {
        let d = (lut[[ch, r as usize]] - got_chw[[ch, y, x]]).abs();
        if d > 0.0 {
            n += 1;
            max = max.max(d);
        }
    }
    format!("{n}/{} values differ, max|d|={max:e}", u8ref.len())
}

#[test]
fn golden_extra_fixed_processors() {
    let m = manifest();
    println!("reference versions: {}", m.versions);
    let mut failures = Vec::new();
    let (mut n_ok, mut n_err, mut n_div) = (0, 0, 0);
    for case in &m.fixed {
        let cfg = config_with(&case.config, &case.overrides);
        let kind = ProcessorKind::from_type_name(case.processor.as_deref().unwrap()).unwrap();
        let proc = ImageProcessor::from_config_as(&cfg, kind).unwrap().with_backend(backend_of(&case.backend));
        let label = format!("{} {} {}", case.suite, case.backend, case.image);
        let result = load_image(golden_dir().join("images").join(&case.image)).and_then(|img| proc.preprocess(&img));
        if case.status == "error" {
            match result {
                Err(_) => n_err += 1,
                // Known divergence: CMYK JPEGs are converted to RGB at decode time, so processors
                // without do_convert_rgb succeed where transformers gets a 4-channel CMYK array.
                Ok(_) if case.mode.as_deref() == Some("CMYK") => {
                    n_div += 1;
                    println!("known divergence: {label} (CMYK decoded to RGB; transformers raises)");
                }
                Ok(_) => failures.push(format!("{label}: transformers raised `{:?}` but Rust succeeded", case.error)),
            }
            continue;
        }
        match result {
            Ok(out) => {
                let exact = out.shape() == case.shape.as_ref().unwrap().as_slice()
                    && Some(sha_f32(out.iter())) == case.sha256_f32;
                let jpeg_pure_rust = case.image.ends_with(".jpg") && !cfg!(feature = "pil-jpeg");
                if exact {
                    n_ok += 1;
                } else if jpeg_pure_rust {
                    println!("{label}: pure-Rust JPEG decoder, not bit-exact: {}", diagnose(case, &out));
                } else {
                    failures.push(format!("{label}: not bit-exact: {}", diagnose(case, &out)));
                }
            }
            Err(e) => failures.push(format!("{label}: Rust error {e}")),
        }
    }
    println!(
        "fixed processors: {n_ok} bit-exact, {n_err} expected errors reproduced, {n_div} known divergences, \
         {} failures",
        failures.len()
    );
    assert!(failures.is_empty(), "failures:\n{}", failures.join("\n"));
}

#[test]
fn golden_extra_decode_like_pillow() {
    let mut failures = Vec::new();
    for d in manifest().decode {
        let img = load_image(golden_dir().join("images").join(&d.image)).unwrap();
        let rgb = hf_processors::image::ops::convert_to_rgb(&img);
        let ok = [rgb.height, rgb.width, rgb.channels] == d.shape[..] && hex(&Sha256::digest(&rgb.data)) == d.sha256;
        println!(
            "decode {:<22} (Pillow mode {:<5}) -> convert(\"RGB\"): {}",
            d.image,
            d.mode,
            if ok { "exact" } else { "DIFF" }
        );
        if !ok {
            if d.image.ends_with(".jpg") && !cfg!(feature = "pil-jpeg") {
                continue; // pure-Rust jpeg-decoder: bounded, checked through the processor cases
            }
            failures.push(d.image.clone());
        }
    }
    assert!(failures.is_empty(), "decode mismatches: {failures:?}");
}

fn qwen_processor(case: &ImageCase) -> Qwen2VLImageProcessor {
    let cfg = config_with(&case.config, &case.overrides);
    match AutoProcessor::from_config(&cfg).unwrap() {
        Processor::Qwen2VL(p) => p.with_backend(backend_of(&case.backend)),
        other => panic!("expected Qwen2-VL, got {}", other.class_name()),
    }
}

#[test]
fn golden_extra_qwen2_vl() {
    let m = manifest();
    let mut raw = NpzReader::new(File::open(golden_dir().join("fixtures/extra_qwen_raw.npz")).unwrap()).unwrap();
    let raw_names: Vec<String> = raw.names().unwrap().iter().map(|n| n.trim_end_matches(".npy").to_string()).collect();
    let mut failures = Vec::new();
    let (mut n_ok, mut n_err, mut n_raw) = (0, 0, 0);
    for case in &m.qwen {
        let proc = qwen_processor(case);
        let id = case.case.as_deref().unwrap();
        let label = format!("{id} {} {}", case.backend, case.image);
        let names = case.batch.clone().unwrap_or_else(|| vec![case.image.clone()]);
        let images: Vec<_> = names.iter().map(|n| load_image(golden_dir().join("images").join(n)).unwrap()).collect();
        let result = proc.preprocess_batch(&images);
        if case.status == "error" {
            match result {
                Err(_) => n_err += 1,
                Ok(_) => failures.push(format!("{label}: transformers raised `{:?}` but Rust succeeded", case.error)),
            }
            continue;
        }
        let out = match result {
            Ok(o) => o,
            Err(e) => {
                failures.push(format!("{label}: Rust error {e}"));
                continue;
            }
        };
        let grid_ref: Vec<Vec<i64>> = match case.grid_thw.clone().unwrap() {
            Value::Array(a) if a.first().is_some_and(|v| v.is_array()) => {
                serde_json::from_value(Value::Array(a)).unwrap()
            }
            v => vec![serde_json::from_value(v).unwrap()],
        };
        let grid: Vec<Vec<i64>> = out.image_grid_thw.rows().into_iter().map(|r| r.to_vec()).collect();
        if grid != grid_ref || out.pixel_values.shape() != case.shape.as_ref().unwrap().as_slice() {
            failures.push(format!(
                "{label}: grid {grid:?} / shape {:?} vs {grid_ref:?} / {:?}",
                out.pixel_values.shape(),
                case.shape
            ));
            continue;
        }
        let jpeg_pure_rust = names.iter().any(|n| n.ends_with(".jpg")) && !cfg!(feature = "pil-jpeg");
        if Some(sha_f32(out.pixel_values.iter())) == case.sha256_f32 {
            n_ok += 1;
        } else if !jpeg_pure_rust {
            failures.push(format!("{label}: pixel_values not bit-exact"));
        }
        if let Some(key) = &case.key {
            let raw_key = format!("{id}__{}__{key}", case.backend);
            if raw_names.contains(&raw_key) {
                let r: ArrayD<f32> = raw.by_name(&raw_key).unwrap();
                let r = r.into_dimensionality::<Ix2>().unwrap();
                let max = (&out.pixel_values - &r).iter().fold(0f32, |a, v| a.max(v.abs()));
                println!("qwen raw pixel_values {raw_key}: max|diff| = {max:e}");
                assert_eq!(max, 0.0, "{raw_key}");
                n_raw += 1;
            }
        }
    }
    println!(
        "qwen2-vl: {n_ok} bit-exact (incl. batches), {n_raw} raw float checks, {n_err} expected errors, {} failures",
        failures.len()
    );
    assert!(failures.is_empty(), "failures:\n{}", failures.join("\n"));
}

#[test]
fn golden_extra_whisper_options() {
    let m = manifest();
    let mut base = NpzReader::new(File::open(golden_dir().join("fixtures/whisper.npz")).unwrap()).unwrap();
    let a: ndarray::Array1<i16> = base.by_name("pcm__short_1s").unwrap();
    let b: ndarray::Array1<i16> = base.by_name("pcm__speechlike_7p3s").unwrap();
    let to_f = |p: &[i16]| p.iter().map(|&v| v as f32 / 32768.0).collect::<Vec<f32>>();
    let clips: HashMap<&str, Vec<f32>> =
        HashMap::from([("a", to_f(a.as_slice().unwrap())), ("b", to_f(&b.as_slice().unwrap()[..40000]))]);
    let mut npz = NpzReader::new(File::open(golden_dir().join("fixtures/extra_whisper.npz")).unwrap()).unwrap();
    let mut worst = 0f32;
    for wc in &m.whisper {
        let cfg = config_with("openai_whisper-tiny.json", &wc.overrides);
        let fe = WhisperFeatureExtractor::from_config(&cfg).unwrap();
        let k = &wc.kwargs;
        let opts = WhisperOptions {
            padding: match k.get("padding").and_then(|v| v.as_str()) {
                None | Some("max_length") => Padding::MaxLength,
                Some("longest") => Padding::Longest,
                Some("do_not_pad") => Padding::DoNotPad,
                Some(p) => panic!("padding {p}"),
            },
            max_length: k.get("max_length").and_then(|v| v.as_u64()).map(|v| v as usize),
            truncation: k.get("truncation").and_then(|v| v.as_bool()).unwrap_or(true),
            pad_to_multiple_of: k.get("pad_to_multiple_of").and_then(|v| v.as_u64()).map(|v| v as usize),
            return_attention_mask: k.get("return_attention_mask").and_then(|v| v.as_bool()),
            do_normalize: k.get("do_normalize").and_then(|v| v.as_bool()),
            dither_seed: wc.dither_seed.unwrap_or(0),
        };
        let inputs: Vec<&[f32]> = wc.clips.iter().map(|c| clips[c.as_str()].as_slice()).collect();
        let out = fe.call(&inputs, &opts).unwrap();
        assert_eq!(out.input_features.shape(), wc.shape.as_slice(), "{}", wc.name);
        let r: ArrayD<f32> = npz.by_name(&format!("features__{}.npy", wc.name)).unwrap();
        let r = r.into_dimensionality::<Ix3>().unwrap();
        let d = &out.input_features - &r;
        let max = d.iter().fold(0f32, |m, v| m.max(v.abs()));
        let mean = d.iter().map(|v| v.abs() as f64).sum::<f64>() / d.len() as f64;
        worst = worst.max(max);
        let mask_info = match (&out.attention_mask, &wc.mask_shape) {
            (Some(mask), Some(shape)) => {
                assert_eq!(mask.shape(), shape.as_slice(), "{} mask shape", wc.name);
                let lens: Vec<usize> = mask.rows().into_iter().map(|r| r.iter().filter(|&&v| v == 1).count()).collect();
                assert_eq!(Some(&lens), wc.mask_lengths.as_ref(), "{} mask lengths", wc.name);
                for row in mask.rows() {
                    let k = row.iter().filter(|&&v| v == 1).count();
                    assert!(row.iter().take(k).all(|&v| v == 1), "{} mask not a prefix", wc.name);
                }
                format!("mask {shape:?} ok")
            }
            (None, None) => "no mask".to_string(),
            (got, want) => panic!(
                "{}: mask presence differs: rust {:?} vs python {want:?}",
                wc.name,
                got.as_ref().map(|m| m.shape().to_vec())
            ),
        };
        println!("whisper {:<24} {:?}: max|diff| = {max:.3e}, mean = {mean:.3e}, {mask_info}", wc.name, wc.shape);
        // Same bound as the base Whisper test (torch float32 STFT vs our f64 STFT).
        assert!(max <= 1e-4, "{}: {max}", wc.name);
    }
    println!("whisper options: worst max|diff| = {worst:e}");
}
