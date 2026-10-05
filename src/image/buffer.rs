//! A minimal interleaved (HWC) 8-bit image buffer, the common currency of the pipeline.
//!
//! Attribution: the CMYK to RGB conversion is a Rust port of Pillow
//! (`src/libImaging/Convert.c`), Copyright (c) 1997-2011 by Secret Labs AB, Copyright (c)
//! 1995-2011 by Fredrik Lundh and contributors, Copyright (c) 2010 by Jeffrey A. Clark and
//! contributors, used under the MIT-CMU (HPND) license. See THIRD_PARTY_NOTICES.md.

use crate::error::{Error, Result};

/// An interleaved 8-bit image: `data[(y * width + x) * channels + c]`.
///
/// `channels` is 1 (L), 2 (LA), 3 (RGB) or 4 (RGBA).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageU8 {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub data: Vec<u8>,
}

impl ImageU8 {
    pub fn new(width: usize, height: usize, channels: usize, data: Vec<u8>) -> Result<Self> {
        if !(1..=4).contains(&channels) {
            return Err(Error::Image(format!("unsupported channel count {channels}")));
        }
        if width == 0 || height == 0 {
            return Err(Error::Image(format!("empty image {width}x{height}")));
        }
        if data.len() != width * height * channels {
            return Err(Error::Image(format!("buffer length {} != {width}x{height}x{channels}", data.len())));
        }
        Ok(Self { width, height, channels, data })
    }

    pub fn zeros(width: usize, height: usize, channels: usize) -> Self {
        Self { width, height, channels, data: vec![0; width * height * channels] }
    }

    #[inline]
    pub fn row(&self, y: usize) -> &[u8] {
        let stride = self.width * self.channels;
        &self.data[y * stride..(y + 1) * stride]
    }

    /// Crop the rectangle `[top, top+h) x [left, left+w)`; the rectangle must lie inside the image.
    pub fn crop(&self, top: usize, left: usize, h: usize, w: usize) -> ImageU8 {
        assert!(top + h <= self.height && left + w <= self.width);
        let c = self.channels;
        let mut data = Vec::with_capacity(h * w * c);
        for y in top..top + h {
            let row = self.row(y);
            data.extend_from_slice(&row[left * c..(left + w) * c]);
        }
        ImageU8 { width: w, height: h, channels: c, data }
    }
}

#[cfg(feature = "decode")]
impl ImageU8 {
    /// Convert from an `image::DynamicImage`, keeping the channel layout (L, LA, RGB, RGBA).
    ///
    /// 16-bit images follow what `PIL.Image.open(...).convert("RGB")` gives for 16-bit PNGs:
    /// grayscale (Pillow mode `I;16`) is clipped to 255 (`min(v, 255)`, not scaled); RGB(A)
    /// and gray+alpha keep the most significant byte (`v >> 8`), with gray+alpha opened as
    /// RGBA like Pillow does. Float images use the `image` crate's rules.
    pub fn from_dynamic(img: &image::DynamicImage) -> ImageU8 {
        use image::DynamicImage as D;
        let (w, h) = (img.width() as usize, img.height() as usize);
        let msb = |v: &[u16]| v.iter().map(|&x| (x >> 8) as u8).collect::<Vec<u8>>();
        match img {
            D::ImageLuma8(b) => ImageU8 { width: w, height: h, channels: 1, data: b.as_raw().clone() },
            D::ImageLumaA8(b) => ImageU8 { width: w, height: h, channels: 2, data: b.as_raw().clone() },
            D::ImageRgb8(b) => ImageU8 { width: w, height: h, channels: 3, data: b.as_raw().clone() },
            D::ImageRgba8(b) => ImageU8 { width: w, height: h, channels: 4, data: b.as_raw().clone() },
            D::ImageLuma16(b) => {
                let data = b.as_raw().iter().map(|&v| v.min(255) as u8).collect();
                ImageU8 { width: w, height: h, channels: 1, data }
            }
            D::ImageLumaA16(b) => {
                let data = b.as_raw().as_chunks::<2>().0.iter().flat_map(|p| {
                    let (l, a) = ((p[0] >> 8) as u8, (p[1] >> 8) as u8);
                    [l, l, l, a]
                });
                ImageU8 { width: w, height: h, channels: 4, data: data.collect() }
            }
            D::ImageRgb16(b) => ImageU8 { width: w, height: h, channels: 3, data: msb(b.as_raw()) },
            D::ImageRgba16(b) => ImageU8 { width: w, height: h, channels: 4, data: msb(b.as_raw()) },
            _ if img.color().has_alpha() => {
                ImageU8 { width: w, height: h, channels: 4, data: img.to_rgba8().into_raw() }
            }
            _ => ImageU8 { width: w, height: h, channels: 3, data: img.to_rgb8().into_raw() },
        }
    }
}

#[cfg(feature = "decode")]
impl From<&image::DynamicImage> for ImageU8 {
    fn from(img: &image::DynamicImage) -> Self {
        ImageU8::from_dynamic(img)
    }
}

#[cfg(feature = "decode")]
impl From<image::DynamicImage> for ImageU8 {
    fn from(img: image::DynamicImage) -> Self {
        ImageU8::from_dynamic(&img)
    }
}

#[cfg(feature = "decode")]
impl From<&image::RgbImage> for ImageU8 {
    fn from(img: &image::RgbImage) -> Self {
        ImageU8 { width: img.width() as usize, height: img.height() as usize, channels: 3, data: img.as_raw().clone() }
    }
}

/// Decode an image file into an [`ImageU8`], keeping the channel layout Pillow would give
/// (L, LA, RGB, RGBA; palettes are expanded).
///
/// JPEG: with the `pil-jpeg` feature the libjpeg-turbo decoder (mozjpeg) is used and pixels
/// are identical to `PIL.Image.open`. Without it, the pure-Rust `jpeg-decoder` is used, which
/// differs from Pillow by up to +-3 per pixel (the `image` crate's default zune-jpeg decoder
/// differs by up to 29, so it is avoided for JPEG).
///
/// The size declared in the image header is checked against
/// [`crate::limits::max_image_pixels`] before anything is decoded (see [`crate::limits`]).
#[cfg(feature = "decode")]
pub fn load_image(path: impl AsRef<std::path::Path>) -> Result<ImageU8> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)?;
    decode_image(&bytes).map_err(|e| match e {
        Error::DecompressionBomb(m) => Error::DecompressionBomb(format!("{}: {m}", path.display())),
        e => Error::Image(format!("{}: {e}", path.display())),
    })
}

/// [`load_image`] for an in-memory encoded image.
#[cfg(feature = "decode")]
pub fn decode_image(bytes: &[u8]) -> Result<ImageU8> {
    #[cfg(feature = "pil-jpeg")]
    return decode_with(bytes, decode_jpeg_libjpeg);
    #[cfg(not(feature = "pil-jpeg"))]
    return decode_with(bytes, decode_jpeg_pure);
}

/// [`decode_image`] with the pure-Rust JPEG decoder, even when `pil-jpeg` is enabled (used by
/// the fuzz targets to cover both decoders in one build).
#[doc(hidden)]
#[cfg(feature = "decode")]
pub fn decode_image_pure_rust(bytes: &[u8]) -> Result<ImageU8> {
    decode_with(bytes, decode_jpeg_pure)
}

#[cfg(feature = "decode")]
type JpegDecoder = fn(&[u8]) -> Result<Option<ImageU8>>;

#[cfg(feature = "decode")]
fn decode_with(bytes: &[u8], jpeg: JpegDecoder) -> Result<ImageU8> {
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF])
        && let Some(img) = jpeg(bytes)?
    {
        return Ok(img);
    }
    decode_generic(bytes)
}

/// PNG (and JPEG pixel formats the JPEG decoders do not handle) through the `image` crate.
#[cfg(feature = "decode")]
fn decode_generic(bytes: &[u8]) -> Result<ImageU8> {
    use image::ImageReader;
    let img_err = |e: image::ImageError| Error::Image(e.to_string());
    let reader = || ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format();
    // Header only: reject bombs before the decoder allocates the pixel buffer.
    let (w, h) = reader()?.into_dimensions().map_err(img_err)?;
    crate::limits::check_image_size(w as u64, h as u64)?;
    let mut r = reader()?;
    let mut limits = image::Limits::no_limits();
    // The `image` crate's own allocation cap (512 MiB by default) would reject images that
    // Pillow accepts; size it for the largest accepted image at 16-bit RGBA instead.
    limits.max_alloc =
        crate::limits::max_image_pixels().map(|p| p.saturating_mul(16).max(512 << 20).saturating_add(64 << 20));
    r.limits(limits);
    let img = r.decode().map_err(img_err)?;
    Ok(ImageU8::from_dynamic(&img))
}

#[cfg(feature = "pil-jpeg")]
fn decode_jpeg_libjpeg(bytes: &[u8]) -> Result<Option<ImageU8>> {
    use mozjpeg::decompress::Format;
    let run = || -> Result<Option<ImageU8>> {
        let io = |e: std::io::Error| Error::Image(format!("jpeg: {e}"));
        // Reads the header only.
        let d = mozjpeg::Decompress::new_mem(bytes).map_err(io)?;
        let (w, h) = (d.width(), d.height());
        crate::limits::check_image_size(w as u64, h as u64)?;
        Ok(match d.image().map_err(io)? {
            Format::RGB(mut s) => {
                let px: Vec<[u8; 3]> = s.read_scanlines().map_err(io)?;
                s.finish().map_err(io)?;
                Some(ImageU8::new(w, h, 3, px.into_flattened())?)
            }
            Format::Gray(mut s) => {
                let px: Vec<u8> = s.read_scanlines().map_err(io)?;
                s.finish().map_err(io)?;
                Some(ImageU8::new(w, h, 1, px)?)
            }
            Format::CMYK(mut s) => {
                let px: Vec<[u8; 4]> = s.read_scanlines().map_err(io)?;
                s.finish().map_err(io)?;
                // Pillow opens CMYK JPEGs with rawmode "CMYK;I" (Adobe inverted storage).
                let data = px.iter().flat_map(|p| cmyk_to_rgb([255 - p[0], 255 - p[1], 255 - p[2], 255 - p[3]]));
                Some(ImageU8::new(w, h, 3, data.collect())?)
            }
        })
    };
    // libjpeg errors unwind out of the C code (mozjpeg's error manager).
    std::panic::catch_unwind(run).unwrap_or_else(|_| Err(Error::Image("jpeg: libjpeg error".into())))
}

#[cfg(feature = "decode")]
fn decode_jpeg_pure(bytes: &[u8]) -> Result<Option<ImageU8>> {
    use jpeg_decoder::PixelFormat;
    let jerr = |e: jpeg_decoder::Error| Error::Image(format!("jpeg: {e}"));
    let mut d = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
    // Header only: reject bombs before the decoder allocates the pixel buffer.
    d.read_info().map_err(jerr)?;
    let info = d.info().ok_or_else(|| Error::Image("jpeg: no frame header".into()))?;
    crate::limits::check_image_size(info.width as u64, info.height as u64)?;
    let data = d.decode().map_err(jerr)?;
    let info = d.info().ok_or_else(|| Error::Image("jpeg: no frame header".into()))?;
    let (w, h) = (info.width as usize, info.height as usize);
    Ok(match info.pixel_format {
        PixelFormat::RGB24 => Some(ImageU8::new(w, h, 3, data)?),
        PixelFormat::L8 => Some(ImageU8::new(w, h, 1, data)?),
        PixelFormat::CMYK32 => {
            // jpeg-decoder already undoes the Adobe inversion.
            let rgb = data.as_chunks::<4>().0.iter().flat_map(|&p| cmyk_to_rgb(p));
            Some(ImageU8::new(w, h, 3, rgb.collect())?)
        }
        _ => None,
    })
}

/// Pillow's `cmyk2rgb` (`Convert.c`): `CLIP8(nk - MULDIV255(c, nk))` with `nk = 255 - k`.
/// CMYK JPEGs are converted to RGB at decode time, as `convert("RGB")` would.
#[cfg(feature = "decode")]
pub(crate) fn cmyk_to_rgb(p: [u8; 4]) -> [u8; 3] {
    let nk = 255 - p[3] as i32;
    let f = |c: u8| {
        let tmp = c as i32 * nk + 128;
        let md = ((tmp >> 8) + tmp) >> 8;
        (nk - md).clamp(0, 255) as u8
    };
    [f(p[0]), f(p[1]), f(p[2])]
}
