//! Decompression-bomb protection, modelled on Pillow's `Image.MAX_IMAGE_PIXELS`.
//!
//! * Decoding an image whose header declares more than [`max_image_pixels`] pixels emits a
//!   warning (see [`set_warning_handler`]), like Pillow's `DecompressionBombWarning`.
//! * More than twice the limit is an [`Error::DecompressionBomb`], like Pillow's
//!   `DecompressionBombError`. The check reads only the image header, so nothing is decoded
//!   or allocated for a rejected image.
//! * Intermediate images and outputs (resize targets, center-crop canvases, Qwen2-VL patch
//!   buffers) are held to the same hard limit (twice [`max_image_pixels`]), so a config with an
//!   absurd `size` or `crop_size` fails with an error instead of trying to allocate it.
//!
//! The default is Pillow's: 89,478,485 pixels (warning) and 178,956,970 pixels (error).
//! `set_max_image_pixels(None)` disables both checks, like `Image.MAX_IMAGE_PIXELS = None`.

use crate::error::{Error, Result};
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Pillow's default `Image.MAX_IMAGE_PIXELS`: `int(1024 * 1024 * 1024 // 4 // 3)`.
pub const DEFAULT_MAX_IMAGE_PIXELS: u64 = 89_478_485;

/// `u64::MAX` encodes "no limit".
static MAX_PIXELS: AtomicU64 = AtomicU64::new(DEFAULT_MAX_IMAGE_PIXELS);

/// A function receiving decompression-bomb warnings.
pub type WarningHandler = fn(&str);

fn stderr_warning(msg: &str) {
    eprintln!("hf_processors: DecompressionBombWarning: {msg}");
}

static WARNING_HANDLER: RwLock<Option<WarningHandler>> = RwLock::new(Some(stderr_warning));

/// The current limit (process-wide); `None` when the checks are disabled.
pub fn max_image_pixels() -> Option<u64> {
    match MAX_PIXELS.load(Ordering::Relaxed) {
        u64::MAX => None,
        v => Some(v),
    }
}

/// Set the process-wide limit. `None` disables the checks (Pillow's
/// `Image.MAX_IMAGE_PIXELS = None`). Images above the limit decode with a warning; images
/// above twice the limit are rejected.
pub fn set_max_image_pixels(limit: Option<u64>) {
    let v = match limit {
        None => u64::MAX,
        Some(v) => v.min(u64::MAX - 1),
    };
    MAX_PIXELS.store(v, Ordering::Relaxed);
}

/// Where warnings go. The default prints one line to stderr; `None` silences them.
pub fn set_warning_handler(handler: Option<WarningHandler>) {
    *WARNING_HANDLER.write().unwrap_or_else(|e| e.into_inner()) = handler;
}

fn warn(msg: &str) {
    let handler = *WARNING_HANDLER.read().unwrap_or_else(|e| e.into_inner());
    if let Some(h) = handler {
        h(msg);
    }
}

/// Pillow's `_decompression_bomb_check` for an image of `width x height` pixels against
/// `limit`: `Err` above `2 * limit`, `Ok(Some(warning))` above `limit`, else `Ok(None)`.
pub fn check_image_size_with(width: u64, height: u64, limit: Option<u64>) -> Result<Option<String>> {
    let Some(limit) = limit else { return Ok(None) };
    let pixels = width as u128 * height as u128;
    let hard = limit as u128 * 2;
    if pixels > hard {
        return Err(Error::DecompressionBomb(format!(
            "image size ({pixels} pixels, {width}x{height}) exceeds limit of {hard} pixels, could be a \
             decompression bomb DOS attack (see hf_processors::limits::set_max_image_pixels)"
        )));
    }
    if pixels > limit as u128 {
        return Ok(Some(format!(
            "image size ({pixels} pixels, {width}x{height}) exceeds limit of {limit} pixels, could be a \
             decompression bomb DOS attack"
        )));
    }
    Ok(None)
}

/// [`check_image_size_with`] against the process-wide limit; a warning is sent to the
/// warning handler.
pub fn check_image_size(width: u64, height: u64) -> Result<()> {
    if let Some(msg) = check_image_size_with(width, height, max_image_pixels())? {
        warn(&msg);
    }
    Ok(())
}

/// The hard limit for a buffer the pipeline is about to allocate: `pixels` (already
/// multiplied by any repeat factor) must not exceed twice the limit.
pub(crate) fn check_alloc(what: &str, dims: &[usize]) -> Result<()> {
    let Some(limit) = max_image_pixels() else { return Ok(()) };
    let pixels = dims.iter().try_fold(1u128, |acc, &d| acc.checked_mul(d as u128)).unwrap_or(u128::MAX);
    if pixels > limit as u128 * 2 {
        return Err(Error::DecompressionBomb(format!(
            "{what} of {dims:?} ({pixels} pixels) exceeds the limit of {} pixels \
             (see hf_processors::limits::set_max_image_pixels)",
            limit as u128 * 2
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pillow_thresholds() {
        let l = Some(DEFAULT_MAX_IMAGE_PIXELS);
        assert!(check_image_size_with(1, DEFAULT_MAX_IMAGE_PIXELS, l).unwrap().is_none());
        assert!(check_image_size_with(1, DEFAULT_MAX_IMAGE_PIXELS + 1, l).unwrap().is_some());
        assert!(check_image_size_with(1, 2 * DEFAULT_MAX_IMAGE_PIXELS, l).unwrap().is_some());
        assert!(matches!(
            check_image_size_with(1, 2 * DEFAULT_MAX_IMAGE_PIXELS + 1, l),
            Err(Error::DecompressionBomb(_))
        ));
        assert!(check_image_size_with(u64::MAX, u64::MAX, None).unwrap().is_none());
        assert!(check_image_size_with(u64::MAX, u64::MAX, l).is_err());
    }
}
