//! WhisperFeatureExtractor on arbitrary f32 audio (NaN, inf, subnormals, empty, long),
//! arbitrary call options and arbitrary extractor parameters.
#![no_main]
use arbitrary::Arbitrary;
use hf_processors::{Padding, WhisperFeatureExtractor, WhisperOptions};
use libfuzzer_sys::fuzz_target;

#[path = "common.rs"]
mod common;

#[derive(Arbitrary, Debug)]
struct Params {
    feature_size: u16,
    sampling_rate: u32,
    hop_length: u16,
    chunk_length: u16,
    n_fft: u16,
    padding_value: f64,
    dither: f64,
}

#[derive(Arbitrary, Debug)]
struct Input<'a> {
    /// None: the openai/whisper-* defaults with a short chunk (to keep runs fast).
    params: Option<Params>,
    chunk_seconds: u8,
    padding: u8,
    max_length: Option<u64>,
    truncation: bool,
    pad_to_multiple_of: Option<u64>,
    return_attention_mask: Option<bool>,
    do_normalize: Option<bool>,
    dither_seed: u64,
    explicit_noise: Option<u32>,
    n_clips: u8,
    audio: &'a [u8],
}

fuzz_target!(|inp: Input| {
    common::init();
    let fe = match &inp.params {
        None => WhisperFeatureExtractor::new(80, 16000, 160, (inp.chunk_seconds % 3) as usize, 400, 0.0),
        Some(p) => WhisperFeatureExtractor::new(
            p.feature_size as usize,
            p.sampling_rate,
            p.hop_length as usize,
            p.chunk_length as usize,
            p.n_fft as usize,
            p.padding_value,
        )
        .map(|mut fe| {
            fe.dither = p.dither;
            fe
        }),
    };
    let Ok(fe) = fe else { return };

    let samples: Vec<f32> = inp.audio.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    let n = 1 + inp.n_clips as usize % 3;
    let per = samples.len() / n;
    let clips: Vec<&[f32]> =
        (0..n).map(|i| &samples[i * per..if i + 1 == n { samples.len() } else { (i + 1) * per }]).collect();
    let opts = WhisperOptions {
        padding: [Padding::MaxLength, Padding::Longest, Padding::DoNotPad][inp.padding as usize % 3],
        max_length: inp.max_length.map(|v| v as usize),
        truncation: inp.truncation,
        pad_to_multiple_of: inp.pad_to_multiple_of.map(|v| v as usize),
        return_attention_mask: inp.return_attention_mask,
        do_normalize: inp.do_normalize,
        dither_seed: inp.dither_seed,
    };

    // Harness bound (not a library limit): each STFT frame costs an n_fft-point FFT plus the dense
    // mel projection (n_bins * n_mels); skip parameter sets that take seconds per run (with
    // hop_length 1 and a long chunk that is millions of FFTs, legitimately slow).
    let longest = clips.iter().map(|c| c.len()).max().unwrap_or(0);
    let padded = opts.max_length.filter(|&m| m > 0).unwrap_or(fe.n_samples).max(longest) as u128
        + opts.pad_to_multiple_of.unwrap_or(0) as u128; // upper bound of the rounding
    let n_fft = fe.n_fft as u128;
    let per_frame = n_fft * (128 - n_fft.leading_zeros() as u128) * 4 + (n_fft / 2 + 1) * fe.feature_size as u128;
    let work = (padded / fe.hop_length as u128 + 1) * per_frame * n as u128;
    // Requests above the pixel limit (2^21 samples here) fail fast in the library: keep those.
    if work > 200_000_000 && padded * n as u128 <= 2 << 20 {
        return;
    }

    let noise: Option<Vec<f32>> = inp.explicit_noise.map(|k| (0..k % 100_000).map(|i| (i as f32).sin()).collect());
    let Ok(out) = fe.call_with_noise(&clips, &opts, noise.as_deref()) else { return };
    let (b, mels, frames) = out.input_features.dim();
    assert_eq!((b, mels), (n, fe.feature_size));
    if let Some(m) = &out.attention_mask {
        assert_eq!(m.nrows(), n);
    }
    // Finite, moderate audio (and dither) gives finite features.
    let finite = samples.iter().all(|x| x.is_finite() && x.abs() < 1e6)
        && fe.dither.is_finite()
        && fe.dither.abs() < 1e6
        && fe.padding_value.is_finite()
        && fe.padding_value.abs() < 1e6;
    if finite && frames > 0 {
        assert!(out.input_features.iter().all(|v| v.is_finite()), "non-finite features from finite audio");
    }
});
