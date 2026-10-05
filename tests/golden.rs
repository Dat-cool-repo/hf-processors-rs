//! Golden tests against Python `transformers` (fixtures from `golden/make_golden.py`).
//!
//! Run with `cargo test --release -- --nocapture` to see the per-case report.

use hf_processors::{Backend, ImageProcessor, PreprocessorConfig, ProcessorKind, WhisperFeatureExtractor, load_image};
use ndarray::{Array2, Array3, ArrayD, Ix2, Ix3};
use ndarray_npy::NpzReader;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

/// (LUTs by channel count, full u8 references by key) for one fixture file.
type FixtureData = (HashMap<usize, Array2<f32>>, HashMap<String, Array3<u8>>);

#[derive(Deserialize)]
struct Manifest {
    versions: serde_json::Value,
    cases: Vec<Case>,
    whisper: Vec<WhisperCase>,
    composite: Vec<CompositeCase>,
}

#[derive(Deserialize)]
struct CompositeCase {
    image: String,
    sha256: String,
}

#[derive(Deserialize, Clone)]
struct Case {
    case: String,
    config: String,
    processor: String,
    backend: String,
    image: String,
    mode: String,
    status: String,
    error: Option<String>,
    shape: Option<Vec<usize>>,
    key: Option<String>,
    fixture: Option<String>,
    sha256: Option<String>,
}

#[derive(Deserialize)]
struct WhisperCase {
    name: String,
    samples: usize,
    shape: Vec<usize>,
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
    let s = std::fs::read_to_string(golden_dir().join("manifest.json")).expect("golden/manifest.json");
    serde_json::from_str(&s).expect("manifest")
}

/// All arrays of an npz, keyed by name without the `.npy` suffix.
fn read_npz_u8(path: &Path) -> HashMap<String, Array3<u8>> {
    let mut out = HashMap::new();
    let Ok(f) = File::open(path) else { return out };
    let mut npz = NpzReader::new(f).expect("npz");
    for name in npz.names().expect("names") {
        let key = name.trim_end_matches(".npy").to_string();
        if key.starts_with("u8__") {
            let a: Array3<u8> = npz.by_name(&name).expect("u8 array");
            out.insert(key, a);
        }
    }
    out
}

fn read_luts(path: &Path) -> HashMap<usize, Array2<f32>> {
    let mut npz = NpzReader::new(File::open(path).expect("fixture npz")).expect("npz");
    let mut out = HashMap::new();
    for name in npz.names().expect("names") {
        let key = name.trim_end_matches(".npy");
        if let Some(c) = key.strip_prefix("lut_c") {
            let a: Array2<f32> = npz.by_name(&name).expect("lut");
            out.insert(c.parse().unwrap(), a);
        }
    }
    out
}

fn hwc_to_chw(img: &hf_processors::ImageU8) -> Array3<u8> {
    let (h, w, c) = (img.height, img.width, img.channels);
    Array3::from_shape_fn((c, h, w), |(ch, y, x)| img.data[(y * w + x) * c + ch])
}

fn backend_of(s: &str) -> Backend {
    match s {
        "pil" => Backend::Pil,
        "torchvision" => Backend::Torchvision,
        other => panic!("unknown backend {other}"),
    }
}

fn build(case: &Case) -> ImageProcessor {
    let cfg = PreprocessorConfig::from_file(golden_dir().join("configs").join(&case.config)).unwrap();
    let kind = ProcessorKind::from_type_name(&case.processor).expect("kind");
    ImageProcessor::from_config_as(&cfg, kind).unwrap().with_backend(backend_of(&case.backend))
}

struct Outcome {
    exact_u8: bool,
    mismatched: usize,
    total: usize,
    max_abs: f32,
    max_u8: i32,
    reference: &'static str,
}

fn run_case(
    case: &Case,
    proc: &ImageProcessor,
    luts: &HashMap<usize, Array2<f32>>,
    full: &HashMap<String, Array3<u8>>,
) -> Result<Outcome, String> {
    let img = load_image(golden_dir().join("images").join(&case.image)).map_err(|e| e.to_string())?;
    let u8_img = proc.preprocess_u8(&img).map_err(|e| e.to_string())?;
    let out = proc.preprocess(&img).map_err(|e| e.to_string())?;
    let chw = hwc_to_chw(&u8_img);
    let shape = case.shape.clone().unwrap();
    if out.shape() != shape.as_slice() {
        return Err(format!("shape {:?} != expected {:?}", out.shape(), shape));
    }
    let digest = hex(&Sha256::digest(chw.as_standard_layout().as_slice().unwrap()));
    let hash_ok = Some(&digest) == case.sha256.as_ref();
    let key = format!("u8__{}", case.key.as_ref().unwrap());
    let (ref_u8, reference) = match full.get(&key) {
        Some(a) => (a.clone(), "full"),
        None if hash_ok => (chw.clone(), "sha256"),
        None => {
            return Err("u8 output differs from reference hash and no full reference is available \
                        (generate them with golden/make_golden.py and set HF_PROCESSORS_GOLDEN_FULL)"
                .into());
        }
    };
    let lut = &luts[&shape[0]];
    let mut max_abs = 0f32;
    let mut mismatched = 0usize;
    let mut max_u8 = 0i32;
    for ((c, y, x), &r) in ref_u8.indexed_iter() {
        let reference = lut[[c, r as usize]];
        let got = out[[c, y, x]];
        max_abs = max_abs.max((reference - got).abs());
        let d = (r as i32 - chw[[c, y, x]] as i32).abs();
        if d != 0 {
            mismatched += 1;
            max_u8 = max_u8.max(d);
        }
    }
    Ok(Outcome { exact_u8: hash_ok, mismatched, total: ref_u8.len(), max_abs, max_u8, reference })
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[test]
fn golden_image_processors() {
    let m = manifest();
    println!("reference versions: {}", m.versions);
    let mut by_fixture: HashMap<String, FixtureData> = HashMap::new();
    let mut failures = Vec::new();
    let mut n_ok = 0;
    let mut n_err = 0;
    let mut worst_png = 0f32;
    let mut jpeg_report = Vec::new();
    println!(
        "{:<28} {:<12} {:<22} {:>6} {:>8} {:>10} {:>7}",
        "case", "backend", "image", "ref", "u8 diff", "max |d|", "result"
    );
    for case in &m.cases {
        let proc = build(case);
        if case.status == "error" {
            let img = load_image(golden_dir().join("images").join(&case.image)).unwrap();
            match proc.preprocess(&img) {
                Err(_) => n_err += 1,
                // Known divergence: without do_convert_rgb, transformers feeds a palette PNG's
                // *indices* (np.array of a mode-"P" image) to the pipeline and then fails on the
                // 1-channel input; the `image` crate expands palettes, so Rust processes colors.
                Ok(_) if case.mode == "P" => {
                    println!(
                        "known divergence: {} {} {} (palette without do_convert_rgb)",
                        case.case, case.backend, case.image
                    );
                }
                Ok(_) => failures.push(format!(
                    "{} {} {} ({}): transformers raised `{}` but Rust succeeded",
                    case.case,
                    case.backend,
                    case.image,
                    case.mode,
                    case.error.as_deref().unwrap_or("")
                )),
            }
            continue;
        }
        let fixture = case.fixture.clone().unwrap();
        let (luts, full) = by_fixture.entry(fixture.clone()).or_insert_with(|| {
            let luts = read_luts(&golden_dir().join("fixtures").join(&fixture));
            let mut full = external_dir().map(|d| read_npz_u8(&d.join(&fixture))).unwrap_or_default();
            full.extend(read_npz_u8(&golden_dir().join("fixtures").join(&fixture)));
            (luts, full)
        });
        // Exactness of the rescale/normalize LUT itself.
        let is_jpeg = case.image.ends_with(".jpg");
        match run_case(case, &proc, luts, full) {
            Ok(o) => {
                let pass = o.max_abs <= 1e-5;
                println!(
                    "{:<28} {:<12} {:<22} {:>6} {:>8} {:>10.3e} {:>7}",
                    case.case,
                    case.backend,
                    case.image,
                    o.reference,
                    format!("{}/{}", o.mismatched, o.max_u8),
                    o.max_abs,
                    if pass {
                        "exact"
                    } else if is_jpeg {
                        "jpeg"
                    } else {
                        "FAIL"
                    }
                );
                if is_jpeg && !cfg!(feature = "pil-jpeg") {
                    jpeg_report.push((case.case.clone(), case.backend.clone(), o.max_abs, o.mismatched, o.total));
                    // Pure-Rust jpeg-decoder vs Pillow's libjpeg-turbo: bounded, not exact.
                    if o.max_abs > 0.1 {
                        failures.push(format!("{} {} jpeg diff too large: {}", case.case, case.backend, o.max_abs));
                    }
                } else {
                    worst_png = worst_png.max(o.max_abs);
                    if !pass || !o.exact_u8 {
                        failures.push(format!(
                            "{} {} {}: max|d|={} u8 mismatches={}",
                            case.case, case.backend, case.image, o.max_abs, o.mismatched
                        ));
                    }
                }
                n_ok += 1;
            }
            Err(e) => failures.push(format!("{} {} {}: {e}", case.case, case.backend, case.image)),
        }
    }
    println!("\n{n_ok} cases compared, {n_err} expected errors reproduced, worst PNG max|diff| = {worst_png:e}");
    for (c, b, d, n, t) in &jpeg_report {
        println!("JPEG decode: {c} {b}: max|diff|={d:.4} ({n}/{t} u8 values differ)");
    }
    assert!(failures.is_empty(), "golden failures:\n{}", failures.join("\n"));
}

/// Independent check on one raw float tensor (no LUT compression involved).
#[test]
fn golden_raw_clip_astronaut() {
    let mut npz =
        NpzReader::new(File::open(golden_dir().join("fixtures/raw_clip_torchvision_astronaut.npz")).unwrap()).unwrap();
    let reference: Array3<f32> = npz.by_name("pixel_values").unwrap();
    let cfg = PreprocessorConfig::from_file(golden_dir().join("configs/openai_clip-vit-base-patch32.json")).unwrap();
    let proc = ImageProcessor::from_config(&cfg).unwrap();
    assert_eq!(proc.backend, Backend::Torchvision);
    let out = proc.preprocess_path(golden_dir().join("images/astronaut.png")).unwrap();
    let max = (&out - &reference).iter().fold(0f32, |m, v| m.max(v.abs()));
    println!("raw CLIP astronaut max|diff| = {max:e}");
    assert!(max <= 1e-5, "max diff {max}");
}

fn lcg_tile(pcm: &[i16], n: usize) -> Vec<i16> {
    pcm.iter().cycle().take(n).copied().collect()
}

#[test]
fn golden_whisper() {
    let m = manifest();
    let cfg = PreprocessorConfig::from_file(golden_dir().join("configs/openai_whisper-tiny.json")).unwrap();
    let fe = WhisperFeatureExtractor::from_config(&cfg).unwrap();
    let mut npz = NpzReader::new(File::open(golden_dir().join("fixtures/whisper.npz")).unwrap()).unwrap();

    // Mel filterbank.
    let mel: ArrayD<f64> = npz.by_name("mel_filters").unwrap();
    let mel = mel.into_dimensionality::<Ix2>().unwrap();
    let n_mels = mel.shape()[1];
    let mut mel_max = 0f64;
    for ((f, j), &v) in mel.indexed_iter() {
        mel_max = mel_max.max((v - fe.mel_filters()[f * n_mels + j]).abs());
    }
    println!("whisper mel filterbank max|diff| = {mel_max:e}");
    assert!(mel_max < 1e-12);

    let short: ndarray::Array1<i16> = npz.by_name("pcm__short_1s").unwrap();
    let base: ndarray::Array1<i16> = npz.by_name("pcm__speechlike_7p3s").unwrap();
    for wc in &m.whisper {
        let pcm: Vec<i16> = match wc.name.as_str() {
            "short_1s" => short.to_vec(),
            "speechlike_7p3s" => base.to_vec(),
            "long_31s" => lcg_tile(base.as_slice().unwrap(), wc.samples),
            other => panic!("unknown clip {other}"),
        };
        assert_eq!(pcm.len(), wc.samples);
        let wave: Vec<f32> = pcm.iter().map(|&v| v as f32 / 32768.0).collect();
        let feats = fe.extract(&wave).unwrap();
        assert_eq!(feats.shape(), wc.shape.as_slice());
        for path in ["torch", "numpy"] {
            let r: ArrayD<f32> = npz.by_name(&format!("{path}__{}", wc.name)).unwrap();
            let r = r.into_dimensionality::<Ix2>().unwrap();
            let d = &feats - &r;
            let max = d.iter().fold(0f32, |m, v| m.max(v.abs()));
            let mean = d.iter().map(|v| v.abs() as f64).sum::<f64>() / d.len() as f64;
            println!("whisper {:<16} vs {path:<5}: max|diff| = {max:.3e}, mean|diff| = {mean:.3e}", wc.name);
            // Python's default (torch) path runs the STFT in float32; see README for the bound.
            let tol = if path == "numpy" { 1e-6 } else { 1e-4 };
            assert!(max <= tol, "{} vs {path}: {max}", wc.name);
        }
    }
    let _ = Ix3;
}

/// Legacy (transformers 4.x) RGB conversion: Pillow's integer alpha compositing on white.
#[test]
fn golden_alpha_composite() {
    for c in manifest().composite {
        let img = load_image(golden_dir().join("images").join(&c.image)).unwrap();
        let rgb = hf_processors::image::ops::convert_to_rgb_composite(&img);
        assert_eq!(hex(&Sha256::digest(&rgb.data)), c.sha256, "{}", c.image);
    }
}
