//! Regression tests for the fuzzing findings (fuzz/) and the decompression-bomb limit: hostile
//! configs, images and audio must give `Err`, never a panic, a hang or a huge allocation.
#![cfg(feature = "decode")]

use hf_processors::{
    AutoProcessor, Backend, Error, ImageU8, Padding, PreprocessorConfig, Processor, Qwen2VLImageProcessor,
    WhisperFeatureExtractor, WhisperOptions, smart_resize,
};
use std::sync::Mutex;

/// The pixel limit is process-wide: tests that depend on it run one at a time.
static LIMIT: Mutex<()> = Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    LIMIT.lock().unwrap_or_else(|e| e.into_inner())
}

fn processor(json: &str) -> hf_processors::Result<Processor> {
    AutoProcessor::from_config(&PreprocessorConfig::from_json_str(json)?)
}

fn image(w: usize, h: usize, c: usize) -> ImageU8 {
    ImageU8::new(w, h, c, (0..w * h * c).map(|i| (i * 31 % 251) as u8).collect()).unwrap()
}

fn run(p: &Processor, img: &ImageU8) -> hf_processors::Result<()> {
    match p {
        Processor::Image(p) => p.preprocess(img).map(drop),
        Processor::Qwen2VL(p) => p.preprocess_batch(std::slice::from_ref(img)).map(drop),
        Processor::Whisper(_) => Ok(()),
    }
}

/// A PNG whose header declares `w x h` but whose only IDAT chunk is empty.
fn png_header_only(w: u32, h: u32) -> Vec<u8> {
    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut chunk = |t: &[u8], d: &[u8]| {
        out.extend((d.len() as u32).to_be_bytes());
        let mut td = t.to_vec();
        td.extend(d);
        out.extend(&td);
        out.extend(crc32(&td).to_be_bytes());
    };
    let mut ihdr = w.to_be_bytes().to_vec();
    ihdr.extend(h.to_be_bytes());
    ihdr.extend([8, 2, 0, 0, 0]);
    chunk(b"IHDR", &ihdr);
    chunk(b"IDAT", &[0x78, 0x9C, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01]); // zlib("")
    chunk(b"IEND", &[]);
    out
}

fn crc32(data: &[u8]) -> u32 {
    let mut c = !0u32;
    for &b in data {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
    }
    !c
}

/// A baseline JPEG declaring `w x h` (SOF0, a quantization table, SOS) with no scan data.
fn jpeg_header_only(w: u16, h: u16) -> Vec<u8> {
    let mut v = vec![0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x11, 0x08];
    v.extend(h.to_be_bytes());
    v.extend(w.to_be_bytes());
    v.extend([3, 1, 0x11, 0, 2, 0x11, 0, 3, 0x11, 0]);
    v.extend([0xFF, 0xDB, 0x00, 0x43, 0x00]); // DQT: table 0, all ones
    v.extend([1u8; 64]);
    v.extend([0xFF, 0xDA, 0x00, 0x0C, 3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0]); // SOS
    v.extend([0xFF, 0xD9]);
    v
}

#[test]
fn decompression_bomb_rejected_from_header() {
    let _g = lock();
    assert_eq!(hf_processors::max_image_pixels(), Some(hf_processors::DEFAULT_MAX_IMAGE_PIXELS));
    // 20000 x 20000 = 4e8 > 2 * 89,478,485: rejected before decoding (the data is missing,
    // so any other outcome would be a decode error instead).
    for bytes in [png_header_only(20000, 20000), jpeg_header_only(20000, 20000)] {
        match hf_processors::decode_image(&bytes) {
            Err(Error::DecompressionBomb(m)) => assert!(m.contains("400000000 pixels"), "{m}"),
            other => panic!("expected DecompressionBomb, got {other:?}"),
        }
        assert!(matches!(
            hf_processors::image::decode_image_pure_rust(&bytes),
            Err(Error::DecompressionBomb(_))
        ));
    }
    // Below the hard limit the header passes and decoding fails normally.
    assert!(matches!(hf_processors::decode_image(&png_header_only(9000, 9000)), Err(Error::Image(_))));
}

#[test]
fn warning_and_error_thresholds() {
    let _g = lock();
    static WARNED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    fn count(_: &str) {
        WARNED.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
    hf_processors::limits::set_warning_handler(Some(count));
    hf_processors::set_max_image_pixels(Some(1000));
    let png = |w: usize, h: usize| {
        let img = image::RgbImage::from_fn(w as u32, h as u32, |x, y| image::Rgb([x as u8, y as u8, 7]));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    };
    let n = |s| WARNED.load(std::sync::atomic::Ordering::SeqCst) - s;
    let s = n(0);
    assert!(hf_processors::decode_image(&png(31, 32)).is_ok());
    assert_eq!(n(s), 0);
    assert!(hf_processors::decode_image(&png(40, 40)).is_ok());
    assert_eq!(n(s), 1);
    assert!(matches!(hf_processors::decode_image(&png(45, 45)), Err(Error::DecompressionBomb(_))));
    // Intermediate buffers: a resize target above twice the limit.
    let clip = processor(r#"{"image_processor_type": "CLIPImageProcessor", "size": {"shortest_edge": 100}}"#).unwrap();
    assert!(matches!(run(&clip, &image(30, 20, 3)), Err(Error::DecompressionBomb(_))));
    hf_processors::set_max_image_pixels(None);
    assert!(hf_processors::decode_image(&png(45, 45)).is_ok());
    assert!(run(&clip, &image(30, 20, 3)).is_ok());
    hf_processors::set_max_image_pixels(Some(hf_processors::DEFAULT_MAX_IMAGE_PIXELS));
    hf_processors::limits::set_warning_handler(None);
}

#[test]
fn absurd_config_sizes_fail_cleanly() {
    let _g = lock();
    let img = image(30, 20, 3);
    for json in [
        r#"{"image_processor_type": "CLIPImageProcessor", "size": {"shortest_edge": 4000000000}}"#,
        r#"{"image_processor_type": "CLIPImageProcessor", "crop_size": {"height": 300000, "width": 300000}}"#,
        r#"{"image_processor_type": "ViTImageProcessor", "size": {"height": 4000000000, "width": 4000000000}}"#,
        r#"{"image_processor_type": "ConvNextImageProcessor", "size": {"shortest_edge": 224}, "crop_pct": 1e-300}"#,
        r#"{"image_processor_type": "SiglipImageProcessor", "size": {"max_height": 4000000000, "max_width": 4000000000}}"#,
        r#"{"image_processor_type": "Qwen2VLImageProcessor", "min_pixels": 1000000000000000, "max_pixels": 10000000000000000}"#,
    ] {
        let p = processor(json).unwrap();
        for b in [Backend::Torchvision, Backend::Pil] {
            let p = match p.clone() {
                Processor::Image(p) => Processor::Image(p.with_backend(b)),
                Processor::Qwen2VL(p) => Processor::Qwen2VL(p.with_backend(b)),
                w => w,
            };
            assert!(matches!(run(&p, &img), Err(Error::DecompressionBomb(_))), "{json}");
        }
    }
    // Out-of-range Qwen patching parameters are config errors.
    for json in [
        r#"{"image_processor_type": "Qwen2VLImageProcessor", "patch_size": 100000}"#,
        r#"{"image_processor_type": "Qwen2VLImageProcessor", "merge_size": 4294967296, "patch_size": 4294967296}"#,
        r#"{"image_processor_type": "Qwen2VLImageProcessor", "temporal_patch_size": 1000}"#,
    ] {
        assert!(matches!(processor(json), Err(Error::Config(_))), "{json}");
    }
}

#[test]
fn mean_std_lengths_are_validated() {
    let img = image(9, 7, 3);
    for (mean, std) in [("[0.5, 0.5]", "[0.5, 0.5, 0.5]"), ("[]", "[0.5]"), ("[0.5]", "[]"), ("[0.1, 0.2, 0.3, 0.4]", "0.5")] {
        let json = format!(r#"{{"image_processor_type": "CLIPImageProcessor", "image_mean": {mean}, "image_std": {std}}}"#);
        let p = processor(&json).unwrap();
        assert!(run(&p, &img).is_err(), "{json}");
    }
    // Without normalization the values are not used (an empty mean used to panic).
    let p = processor(r#"{"image_processor_type": "CLIPImageProcessor", "image_mean": [], "do_normalize": false}"#).unwrap();
    assert!(run(&p, &img).is_ok());
    // NaN / zero / negative statistics are not errors (transformers divides anyway).
    let mut cfg = PreprocessorConfig::from_json_str(r#"{"image_processor_type": "CLIPImageProcessor"}"#).unwrap();
    cfg.image_std = Some(hf_processors::config::FloatOrVec::Vec(vec![0.0, -1.0, f64::NAN]));
    cfg.rescale_factor = Some(f64::INFINITY);
    let p = AutoProcessor::from_config(&cfg).unwrap();
    assert!(run(&p, &img).is_ok());
}

#[test]
fn smart_resize_extremes() {
    // Python ints are unbounded; sizes that do not fit are errors, not overflows.
    assert!(smart_resize(usize::MAX, usize::MAX, 28, 3136, 12845056).is_err());
    assert!(smart_resize(1 << 40, 1 << 40, 28, 3136, 12845056).is_err());
    assert!(smart_resize(1, 1, usize::MAX, 0, usize::MAX).is_err());
    // A huge min_pixels gives a huge (but representable) size; the processor then refuses it.
    let (h, w) = smart_resize(100, 100, 28, usize::MAX, usize::MAX).unwrap();
    assert!(h > 1 << 31 && w > 1 << 31);
    assert_eq!(smart_resize(1, 200, 28, 3136, 12845056).unwrap(), (28, 812));
    assert!(smart_resize(1, 201, 28, 3136, 12845056).is_err());
    assert_eq!(smart_resize(1_000_000, 1_000_000, 28, 3136, 0).unwrap(), (28, 28));
    let (h, w) = smart_resize(4_000_000_000, 4_000_000_000, 28, 3136, 12845056).unwrap();
    assert!(h % 28 == 0 && w % 28 == 0 && h * w <= 12845056 && h >= 3500, "{h}x{w}");
    // A zero budget rounds a 5x3 image to 0x0, which cannot be resized (transformers fails too).
    let q = Qwen2VLImageProcessor::from_json_str(r#"{"min_pixels": 0, "max_pixels": 0}"#).unwrap();
    assert!(q.preprocess(&image(5, 3, 3)).is_err());
}

#[test]
fn whisper_odd_n_fft_and_bad_parameters() {
    let _g = lock();
    // Odd n_fft: transformers keeps (n - 1) // hop frames; this used to panic in ndarray.
    let fe = WhisperFeatureExtractor::new(80, 16000, 160, 1, 401, 0.0).unwrap();
    let x: Vec<f32> = (0..1600).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
    let opts = WhisperOptions { padding: Padding::Longest, return_attention_mask: Some(true), ..Default::default() };
    let out = fe.call(&[&x], &opts).unwrap();
    assert_eq!(out.input_features.dim(), (1, 80, 9));
    assert_eq!(out.attention_mask.unwrap().dim(), (1, 10));
    assert_eq!(fe.extract(&x).unwrap().dim(), (80, 99));
    for (fs, sr, hop, chunk, n_fft) in [
        (80, 16000, 0, 30, 400),
        (80, 0, 160, 30, 400),
        (0, 16000, 160, 30, 400),
        (80, 16000, 160, 30, 0),
        (80, 16000, 160, 30, 1 << 20),
        (100_000, 16000, 160, 30, 400),
        (80, u32::MAX, 160, usize::MAX, 400),
    ] {
        assert!(WhisperFeatureExtractor::new(fs, sr, hop, chunk, n_fft, 0.0).is_err());
    }
    // Overflowing / absurd padding requests.
    let fe = WhisperFeatureExtractor::new(80, 16000, 160, 30, 400, 0.0).unwrap();
    for opts in [
        WhisperOptions { pad_to_multiple_of: Some(usize::MAX), max_length: Some(usize::MAX - 1), ..Default::default() },
        WhisperOptions { max_length: Some(1 << 40), ..Default::default() },
    ] {
        assert!(fe.call(&[&x], &opts).is_err());
    }
    // NaN / inf / empty audio.
    let bad = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MIN_POSITIVE / 2.0, f32::MAX];
    let clip: Vec<f32> = bad.iter().cycle().take(5000).copied().collect();
    assert!(fe.extract(&clip).is_ok());
    assert!(fe.extract(&[]).is_ok());
    let opts = WhisperOptions { padding: Padding::Longest, ..Default::default() };
    assert!(fe.call(&[&[]], &opts).is_err());
}

#[test]
fn hostile_inputs_never_panic() {
    let _g = lock();
    // Truncated / garbage image bytes.
    for bytes in [&b""[..], b"\xFF\xD8\xFF", b"\x89PNG\r\n\x1a\n", b"GIF89a", &[0u8; 64]] {
        assert!(hf_processors::decode_image(bytes).is_err());
        assert!(hf_processors::image::decode_image_pure_rust(bytes).is_err());
    }
    // Zero-sized crops and resize targets.
    let p = processor(r#"{"image_processor_type": "CLIPImageProcessor", "crop_size": {"height": 0, "width": 0}}"#).unwrap();
    assert!(run(&p, &image(10, 10, 3)).is_err());
    let p = processor(r#"{"image_processor_type": "CLIPImageProcessor", "size": {"shortest_edge": 0}}"#).unwrap();
    assert!(run(&p, &image(10, 10, 3)).is_err());
}
