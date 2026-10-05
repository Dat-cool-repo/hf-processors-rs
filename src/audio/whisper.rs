//! `transformers.WhisperFeatureExtractor`: pad/truncate (to 30 s by default), optional
//! zero-mean/unit-variance normalization, optional dither, STFT (periodic Hann,
//! reflect-padded, `center=True`), power spectrum, Slaney mel filterbank, `log10`,
//! dynamic-range clamp (`max - 8`) and `(x + 4) / 4`.
//!
//! The waveform-side steps (padding, normalization, dither) use the same float32 arithmetic
//! as transformers. The spectrogram is computed in `f64` and returned as `f32`; Python's
//! default path (torch installed) runs the STFT in `f32`, so the features agree to float32
//! noise (about 2e-5), not bit-for-bit (see README).
//!
//! Attribution: the behaviour (including the Slaney mel filterbank) is reimplemented from
//! Hugging Face `transformers` (`WhisperFeatureExtractor`, `audio_utils.py`), Copyright The
//! HuggingFace Team, Apache-2.0 license. See THIRD_PARTY_NOTICES.md.

use crate::config::PreprocessorConfig;
use crate::error::{Error, Result};
use ndarray::{Array2, Array3, Axis};
use rustfft::{FftPlanner, num_complex::Complex};
use std::sync::Arc;

#[derive(Clone)]
pub struct WhisperFeatureExtractor {
    pub feature_size: usize,
    pub sampling_rate: u32,
    pub hop_length: usize,
    pub chunk_length: usize,
    pub n_fft: usize,
    pub padding_value: f64,
    pub n_samples: usize,
    pub nb_max_frames: usize,
    /// Standard deviation of the Gaussian noise added to the padded waveform (0 = off).
    pub dither: f64,
    /// The config's `return_attention_mask` (used when a call does not specify it).
    pub return_attention_mask: bool,
    /// `(1 + n_fft / 2, feature_size)`, row-major, as `transformers` stores it.
    mel_filters: Vec<f64>,
    window: Vec<f64>,
    fft: Arc<dyn rustfft::Fft<f64>>,
}

impl std::fmt::Debug for WhisperFeatureExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WhisperFeatureExtractor")
            .field("feature_size", &self.feature_size)
            .field("sampling_rate", &self.sampling_rate)
            .field("hop_length", &self.hop_length)
            .field("chunk_length", &self.chunk_length)
            .field("n_fft", &self.n_fft)
            .field("dither", &self.dither)
            .finish()
    }
}

/// `padding=` of `WhisperFeatureExtractor.__call__`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Padding {
    /// `"max_length"` (the default): pad to `max_length` (default `n_samples`, i.e. 30 s).
    #[default]
    MaxLength,
    /// `"longest"` / `True`: pad to the longest (truncated) clip of the batch.
    Longest,
    /// `"do_not_pad"` / `False`: no padding (a batch then needs equal lengths).
    DoNotPad,
}

/// Keyword arguments of `WhisperFeatureExtractor.__call__`.
#[derive(Clone, Debug)]
pub struct WhisperOptions {
    pub padding: Padding,
    /// `max_length` in samples; `None` (or 0) means `n_samples`.
    pub max_length: Option<usize>,
    /// Cut clips longer than `max_length` (default `true`).
    pub truncation: bool,
    pub pad_to_multiple_of: Option<usize>,
    /// `Some(true)`: return the frame-level mask `(N, frames)`. `None`/`Some(false)`: see
    /// [`WhisperFeatures::attention_mask`] for transformers' sample-level quirk.
    pub return_attention_mask: Option<bool>,
    /// Zero-mean/unit-variance normalization of each clip's unpadded samples.
    pub do_normalize: Option<bool>,
    /// Seed of the noise used when `dither != 0` (see [`gaussian_noise`]).
    pub dither_seed: u64,
}

impl Default for WhisperOptions {
    fn default() -> Self {
        WhisperOptions {
            padding: Padding::MaxLength,
            max_length: None,
            truncation: true,
            pad_to_multiple_of: None,
            return_attention_mask: None,
            do_normalize: None,
            dither_seed: 0,
        }
    }
}

/// Result of [`WhisperFeatureExtractor::call`].
#[derive(Clone, Debug)]
pub struct WhisperFeatures {
    /// `(N, feature_size, frames)` with `frames = padded_length / hop_length`.
    pub input_features: Array3<f32>,
    /// With `return_attention_mask=Some(true)`: `(N, frames)`, 1 for frames of real audio.
    ///
    /// transformers quirk, reproduced: when the mask was only computed for padding (the call
    /// passed `do_normalize=True`, or left `return_attention_mask` unset while the config sets
    /// it), it is returned at *sample* level, `(N, padded_length)`.
    pub attention_mask: Option<Array2<i32>>,
}

/// Deterministic standard-normal noise: SplitMix64 (counter `seed + i * 0x9E3779B97F4A7C15`,
/// `i = 1, 2, ...`) feeding Box-Muller in `f64`, rounded to `f32`. Pairs `(u1, u2)` give
/// `r * cos(2 pi u2), r * sin(2 pi u2)` with `r = sqrt(-2 ln u1)`, `u1 = ((x >> 11) + 1) / 2^53`,
/// `u2 = (x >> 11) / 2^53`. Not torch's `randn` stream: pass torch's noise explicitly to
/// [`WhisperFeatureExtractor::call_with_noise`] to reproduce a seeded torch run.
pub fn gaussian_noise(seed: u64, n: usize) -> Vec<f32> {
    const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
    let mix = |i: u64| {
        let mut z = seed.wrapping_add(i.wrapping_mul(GOLDEN));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let scale = 1.0 / (1u64 << 53) as f64;
    let mut out = Vec::with_capacity(n + 1);
    for k in 0..n.div_ceil(2) as u64 {
        let u1 = ((mix(2 * k + 1) >> 11) + 1) as f64 * scale;
        let u2 = (mix(2 * k + 2) >> 11) as f64 * scale;
        let r = (-2.0 * u1.ln()).sqrt();
        let th = 2.0 * std::f64::consts::PI * u2;
        out.push((r * th.cos()) as f32);
        out.push((r * th.sin()) as f32);
    }
    out.truncate(n);
    out
}

/// `audio_utils.hertz_to_mel(..., mel_scale="slaney")`.
fn hertz_to_mel_slaney(freq: f64) -> f64 {
    let min_log_hertz = 1000.0;
    let min_log_mel = 15.0;
    let logstep = 27.0 / 6.4f64.ln();
    if freq >= min_log_hertz { min_log_mel + (freq / min_log_hertz).ln() * logstep } else { 3.0 * freq / 200.0 }
}

/// `audio_utils.mel_to_hertz(..., mel_scale="slaney")`.
fn mel_to_hertz_slaney(mels: f64) -> f64 {
    let min_log_hertz = 1000.0;
    let min_log_mel = 15.0;
    let logstep = 6.4f64.ln() / 27.0;
    if mels >= min_log_mel { min_log_hertz * (logstep * (mels - min_log_mel)).exp() } else { 200.0 * mels / 3.0 }
}

/// `np.linspace(start, stop, num)` (numpy's formula: `start + i * step`, last = stop).
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 1 {
        return vec![start];
    }
    let div = (num - 1) as f64;
    let step = (stop - start) / div;
    let mut v: Vec<f64> = (0..num).map(|i| start + i as f64 * step).collect();
    v[num - 1] = stop;
    v
}

/// `audio_utils.mel_filter_bank(norm="slaney", mel_scale="slaney")`, shape `(n_freqs, n_mels)`.
pub fn mel_filter_bank_slaney(
    num_frequency_bins: usize,
    num_mel_filters: usize,
    min_frequency: f64,
    max_frequency: f64,
    sampling_rate: u32,
) -> Vec<f64> {
    let mel_min = hertz_to_mel_slaney(min_frequency);
    let mel_max = hertz_to_mel_slaney(max_frequency);
    let mel_freqs = linspace(mel_min, mel_max, num_mel_filters + 2);
    let filter_freqs: Vec<f64> = mel_freqs.iter().map(|&m| mel_to_hertz_slaney(m)).collect();
    let fft_freqs = linspace(0.0, (sampling_rate / 2) as f64, num_frequency_bins);
    let filter_diff: Vec<f64> = filter_freqs.windows(2).map(|w| w[1] - w[0]).collect();
    let mut out = vec![0f64; num_frequency_bins * num_mel_filters];
    for (i, &f) in fft_freqs.iter().enumerate() {
        for m in 0..num_mel_filters {
            // slopes[i, j] = filter_freqs[j] - fft_freqs[i]
            let down = -(filter_freqs[m] - f) / filter_diff[m];
            let up = (filter_freqs[m + 2] - f) / filter_diff[m + 1];
            let v = 0f64.max(down.min(up));
            let enorm = 2.0 / (filter_freqs[m + 2] - filter_freqs[m]);
            out[i * num_mel_filters + m] = v * enorm;
        }
    }
    out
}

/// Round `n` up to a multiple of `m` (transformers' `pad_to_multiple_of`).
fn round_up(n: usize, m: Option<usize>) -> usize {
    match m {
        Some(m) if m > 0 && !n.is_multiple_of(m) => (n / m + 1) * m,
        _ => n,
    }
}

impl WhisperFeatureExtractor {
    pub fn new(
        feature_size: usize,
        sampling_rate: u32,
        hop_length: usize,
        chunk_length: usize,
        n_fft: usize,
        padding_value: f64,
    ) -> Self {
        let n_samples = chunk_length * sampling_rate as usize;
        let mel_filters = mel_filter_bank_slaney(1 + n_fft / 2, feature_size, 0.0, 8000.0, sampling_rate);
        // np.hanning(n_fft + 1)[:-1] == torch.hann_window(n_fft, periodic=True)
        let window = (0..n_fft)
            .map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / n_fft as f64).cos())
            .collect();
        let fft = FftPlanner::new().plan_fft_forward(n_fft);
        WhisperFeatureExtractor {
            feature_size,
            sampling_rate,
            hop_length,
            chunk_length,
            n_fft,
            padding_value,
            n_samples,
            nb_max_frames: n_samples / hop_length,
            dither: 0.0,
            return_attention_mask: false,
            mel_filters,
            window,
            fft,
        }
    }

    pub fn from_config(cfg: &PreprocessorConfig) -> Result<Self> {
        let mut fe = Self::new(
            cfg.feature_size.unwrap_or(80),
            cfg.sampling_rate.unwrap_or(16000),
            cfg.hop_length.unwrap_or(160),
            cfg.chunk_length.unwrap_or(30),
            cfg.n_fft.unwrap_or(400),
            cfg.padding_value.unwrap_or(0.0),
        );
        fe.dither = cfg.dither.unwrap_or(0.0);
        fe.return_attention_mask = cfg.return_attention_mask.unwrap_or(false);
        Ok(fe)
    }

    /// The `(n_freqs, n_mels)` filterbank (row-major).
    pub fn mel_filters(&self) -> &[f64] {
        &self.mel_filters
    }

    /// Log-mel features for one clip (mono, at `sampling_rate`), padded/truncated to
    /// `n_samples`: shape `(feature_size, nb_max_frames)`, i.e. `input_features[0]`.
    pub fn extract(&self, raw: &[f32]) -> Result<Array2<f32>> {
        let out = self.call(&[raw], &WhisperOptions::default())?;
        Ok(out.input_features.index_axis_move(Axis(0), 0))
    }

    /// Batch version with default options, `(N, feature_size, nb_max_frames)`.
    pub fn extract_batch(&self, clips: &[&[f32]]) -> Result<Array3<f32>> {
        Ok(self.call(clips, &WhisperOptions::default())?.input_features)
    }

    /// `WhisperFeatureExtractor.__call__(clips, sampling_rate=..., return_tensors="np", **opts)`.
    /// Dither noise (when `self.dither != 0`) comes from [`gaussian_noise`]`(opts.dither_seed)`.
    pub fn call(&self, clips: &[&[f32]], opts: &WhisperOptions) -> Result<WhisperFeatures> {
        self.call_with_noise(clips, opts, None)
    }

    /// Like [`call`](Self::call), with explicit standard-normal dither noise: `noise` holds
    /// `N * padded_length` values, row-major, exactly what `torch.randn(waveform.shape)` returns
    /// inside transformers (it is scaled by `dither` here). Ignored when `dither == 0`.
    pub fn call_with_noise(
        &self,
        clips: &[&[f32]],
        opts: &WhisperOptions,
        noise: Option<&[f32]>,
    ) -> Result<WhisperFeatures> {
        if clips.is_empty() {
            return Err(Error::Audio("empty batch".into()));
        }
        let max_length = opts.max_length.filter(|&m| m > 0).unwrap_or(self.n_samples);
        // `return_attention_mask or do_normalize`, then the config default when that is None.
        let pad_arg = if opts.return_attention_mask == Some(true) { Some(true) } else { opts.do_normalize };
        let make_mask = pad_arg.unwrap_or(self.return_attention_mask);

        // Truncation (SequenceFeatureExtractor._truncate).
        let trunc_len = round_up(max_length, opts.pad_to_multiple_of);
        let lens: Vec<usize> =
            clips.iter().map(|c| if opts.truncation { c.len().min(trunc_len) } else { c.len() }).collect();
        // Padding target (SequenceFeatureExtractor._pad).
        let target = match opts.padding {
            Padding::DoNotPad => None,
            Padding::MaxLength => Some(round_up(max_length, opts.pad_to_multiple_of)),
            Padding::Longest => Some(round_up(*lens.iter().max().unwrap(), opts.pad_to_multiple_of)),
        };
        let padded_lens: Vec<usize> = lens.iter().map(|&l| target.map_or(l, |t| t.max(l))).collect();
        let total = padded_lens[0];
        if padded_lens.iter().any(|&l| l != total) {
            return Err(Error::Audio(format!(
                "clips have different lengths after padding ({padded_lens:?}); use Padding::Longest or MaxLength"
            )));
        }
        if total <= self.n_fft / 2 {
            return Err(Error::Audio(format!(
                "padded waveform has {total} samples; reflect padding needs more than n_fft / 2 = {}",
                self.n_fft / 2
            )));
        }
        let n = clips.len();
        let pad_value = self.padding_value as f32;
        let mut waves: Vec<Vec<f32>> = Vec::with_capacity(n);
        for (clip, &len) in clips.iter().zip(&lens) {
            let mut w = Vec::with_capacity(total);
            w.extend_from_slice(&clip[..len]);
            w.resize(total, pad_value);
            if opts.do_normalize == Some(true) {
                normalize_in_place(&mut w, len, pad_value);
            }
            waves.push(w);
        }
        if self.dither != 0.0 {
            let generated;
            let noise = match noise {
                Some(z) => z,
                None => {
                    generated = gaussian_noise(opts.dither_seed, n * total);
                    &generated
                }
            };
            if noise.len() != n * total {
                return Err(Error::Audio(format!("noise has {} values, expected {}", noise.len(), n * total)));
            }
            let d = self.dither as f32;
            for (w, z) in waves.iter_mut().zip(noise.chunks_exact(total)) {
                for (x, &e) in w.iter_mut().zip(z) {
                    *x += d * e;
                }
            }
        }

        let frames = total / self.hop_length;
        let mut feats = Array3::<f32>::zeros((n, self.feature_size, frames));
        for (i, w) in waves.iter().enumerate() {
            let wave: Vec<f64> = w.iter().map(|&v| v as f64).collect();
            let log_spec = self.log_mel(&wave)?;
            let max = log_spec.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let floor = max - 8.0;
            feats.index_axis_mut(Axis(0), i).assign(&log_spec.mapv(|v| ((v.max(floor) + 4.0) / 4.0) as f32));
        }

        let attention_mask = if opts.return_attention_mask == Some(true) {
            // mask[:, ::hop_length], minus the last entry when total % hop_length != 0.
            Some(Array2::from_shape_fn((n, frames), |(i, f)| (f * self.hop_length < lens[i]) as i32))
        } else if make_mask {
            Some(Array2::from_shape_fn((n, total), |(i, s)| (s < lens[i]) as i32))
        } else {
            None
        };
        Ok(WhisperFeatures { input_features: feats, attention_mask })
    }

    /// log10(max(mel @ |STFT|^2, 1e-10)) without the final frame, shape `(n_mels, frames)`.
    fn log_mel(&self, wave: &[f64]) -> Result<Array2<f64>> {
        let n_fft = self.n_fft;
        let pad = n_fft / 2;
        if wave.len() <= pad {
            return Err(Error::Audio("waveform shorter than n_fft / 2".into()));
        }
        // Reflect padding (numpy "reflect" / torch "reflect": edge sample not repeated).
        let n = wave.len();
        let mut padded = Vec::with_capacity(n + 2 * pad);
        for i in (1..=pad).rev() {
            padded.push(wave[i]);
        }
        padded.extend_from_slice(wave);
        for i in 0..pad {
            padded.push(wave[n - 2 - i]);
        }
        let num_frames = 1 + (padded.len() - n_fft) / self.hop_length;
        let frames = num_frames - 1; // log_spec[:, :-1]
        let n_bins = n_fft / 2 + 1;
        let n_mels = self.feature_size;
        let mut power = vec![0f64; n_bins];
        let mut buf = vec![Complex::new(0f64, 0f64); n_fft];
        let mut scratch = vec![Complex::new(0f64, 0f64); self.fft.get_inplace_scratch_len()];
        let mut out = Array2::<f64>::zeros((n_mels, frames));
        for t in 0..frames {
            let start = t * self.hop_length;
            for (k, b) in buf.iter_mut().enumerate() {
                *b = Complex::new(padded[start + k] * self.window[k], 0.0);
            }
            self.fft.process_with_scratch(&mut buf, &mut scratch);
            for (p, c) in power.iter_mut().zip(buf.iter()) {
                *p = c.re * c.re + c.im * c.im;
            }
            for m in 0..n_mels {
                let mut s = 0f64;
                for (f, &p) in power.iter().enumerate() {
                    s += self.mel_filters[f * n_mels + m] * p;
                }
                out[[m, t]] = s.max(1e-10).log10();
            }
        }
        Ok(out)
    }
}

/// `zero_mean_unit_var_norm` for one clip: statistics over the first `len` samples (computed
/// in f64, then rounded to f32 like numpy's float32 results), applied in f32; the padded tail
/// is reset to `pad_value`.
fn normalize_in_place(w: &mut [f32], len: usize, pad_value: f32) {
    let valid = &w[..len];
    let n = len.max(1) as f64;
    let mean64 = valid.iter().map(|&x| x as f64).sum::<f64>() / n;
    let mean = mean64 as f32;
    let var = (valid.iter().map(|&x| (x as f64 - mean64).powi(2)).sum::<f64>() / n) as f32;
    let denom = (var + 1e-7f32).sqrt();
    for x in w[..len].iter_mut() {
        *x = (*x - mean) / denom;
    }
    for x in w[len..].iter_mut() {
        *x = pad_value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes() {
        let fe = WhisperFeatureExtractor::new(80, 16000, 160, 30, 400, 0.0);
        let x: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        let f = fe.extract(&x).unwrap();
        assert_eq!(f.dim(), (80, 3000));
    }

    #[test]
    fn padding_modes() {
        let fe = WhisperFeatureExtractor::new(80, 16000, 160, 30, 400, 0.0);
        let a: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.05).sin()).collect();
        let b: Vec<f32> = (0..1650).map(|i| (i as f32 * 0.03).sin()).collect();
        let opts = WhisperOptions {
            padding: Padding::Longest,
            return_attention_mask: Some(true),
            ..Default::default()
        };
        let out = fe.call(&[&a, &b], &opts).unwrap();
        assert_eq!(out.input_features.dim(), (2, 80, 10));
        let m = out.attention_mask.unwrap();
        assert_eq!(m.dim(), (2, 10));
        assert_eq!(m.row(0).sum(), 7); // ceil(1000 / 160)
        assert_eq!(m.row(1).sum(), 10);
        let opts = WhisperOptions { padding: Padding::DoNotPad, ..Default::default() };
        assert!(fe.call(&[&a, &b], &opts).is_err());
    }

    #[test]
    fn noise_is_standard_normal() {
        let z = gaussian_noise(7, 100_001);
        assert_eq!(z.len(), 100_001);
        let mean = z.iter().map(|&v| v as f64).sum::<f64>() / z.len() as f64;
        let var = z.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / z.len() as f64;
        assert!(mean.abs() < 0.01 && (var - 1.0).abs() < 0.02, "{mean} {var}");
        assert_eq!(gaussian_noise(7, 10), z[..10]);
    }
}
