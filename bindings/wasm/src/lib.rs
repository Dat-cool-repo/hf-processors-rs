//! In-browser preprocessing: `wasm-pack build bindings/wasm --target web` (or plain
//! `cargo build --target wasm32-unknown-unknown` + `wasm-bindgen`).
//!
//! ```js
//! const p = new Preprocessor(await (await fetch("preprocessor_config.json")).text(), "torchvision");
//! const { data, width, height } = ctx.getImageData(0, 0, w, h);
//! const out = p.preprocessRgba(data, width, height);   // out.data: Float32Array, out.shape: [1, 3, 224, 224]
//! ```

use hf_processors::{AutoProcessor, Backend, ImageU8, PreprocessorConfig, Processor, WhisperOptions, decode_image};
use wasm_bindgen::prelude::*;

fn js_err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

/// Pixel limit for `preprocessEncoded` (Pillow's `MAX_IMAGE_PIXELS` semantics; the default is
/// 89,478,485): images above twice the limit are rejected from their header, before decoding.
/// `undefined` / `null` disables the check.
#[wasm_bindgen(js_name = setMaxImagePixels)]
pub fn set_max_image_pixels(limit: Option<f64>) {
    hf_processors::set_max_image_pixels(limit.map(|v| v.max(0.0) as u64));
}

/// A tensor returned to JavaScript: flat row-major `data` plus its `shape`.
#[wasm_bindgen]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<u32>,
    grid_thw: Option<Vec<i32>>,
}

#[wasm_bindgen]
impl Tensor {
    /// The values (copied into a `Float32Array`).
    #[wasm_bindgen(getter)]
    pub fn data(&self) -> Vec<f32> {
        self.data.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn shape(&self) -> Vec<u32> {
        self.shape.clone()
    }

    /// Qwen2-VL only: `[t, h, w]` per image, flattened.
    #[wasm_bindgen(getter, js_name = gridThw)]
    pub fn grid_thw(&self) -> Option<Vec<i32>> {
        self.grid_thw.clone()
    }
}

/// Any supported processor, built from the text of a `preprocessor_config.json`.
#[wasm_bindgen]
pub struct Preprocessor {
    inner: Processor,
}

#[wasm_bindgen]
impl Preprocessor {
    /// `backend`: `"torchvision"` (default; transformers v5's default classes) or `"pil"`.
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str, backend: Option<String>) -> Result<Preprocessor, JsError> {
        let cfg = PreprocessorConfig::from_json_str(config_json).map_err(js_err)?;
        let mut inner = AutoProcessor::from_config(&cfg).map_err(js_err)?;
        let backend = match backend.as_deref() {
            None | Some("torchvision") => Backend::Torchvision,
            Some("pil") => Backend::Pil,
            Some(other) => return Err(JsError::new(&format!("unknown backend {other}"))),
        };
        match &mut inner {
            Processor::Image(p) => p.backend = backend,
            Processor::Qwen2VL(p) => p.backend = backend,
            Processor::Whisper(_) => {}
        }
        Ok(Preprocessor { inner })
    }

    /// The transformers class reproduced.
    #[wasm_bindgen(getter, js_name = className)]
    pub fn class_name(&self) -> String {
        self.inner.class_name().to_string()
    }

    /// RGBA pixels as returned by `CanvasRenderingContext2D.getImageData`.
    #[wasm_bindgen(js_name = preprocessRgba)]
    pub fn preprocess_rgba(&self, rgba: &[u8], width: u32, height: u32) -> Result<Tensor, JsError> {
        let img = ImageU8::new(width as usize, height as usize, 4, rgba.to_vec()).map_err(js_err)?;
        self.run(img)
    }

    /// Interleaved pixels with `channels` = 1 (L), 3 (RGB) or 4 (RGBA).
    #[wasm_bindgen(js_name = preprocessPixels)]
    pub fn preprocess_pixels(&self, data: &[u8], width: u32, height: u32, channels: u32) -> Result<Tensor, JsError> {
        let img = ImageU8::new(width as usize, height as usize, channels as usize, data.to_vec()).map_err(js_err)?;
        self.run(img)
    }

    /// An encoded PNG / JPEG file.
    #[wasm_bindgen(js_name = preprocessEncoded)]
    pub fn preprocess_encoded(&self, bytes: &[u8]) -> Result<Tensor, JsError> {
        self.run(decode_image(bytes).map_err(js_err)?)
    }

    /// Whisper: mono samples at the extractor's sampling rate -> log-mel `(1, n_mels, frames)`.
    #[wasm_bindgen(js_name = extractAudio)]
    pub fn extract_audio(&self, samples: &[f32]) -> Result<Tensor, JsError> {
        let Processor::Whisper(fe) = &self.inner else {
            return Err(JsError::new("not an audio feature extractor"));
        };
        let out = fe.call(&[samples], &WhisperOptions::default()).map_err(js_err)?;
        let shape = out.input_features.shape().iter().map(|&d| d as u32).collect();
        Ok(Tensor { data: out.input_features.into_raw_vec_and_offset().0, shape, grid_thw: None })
    }

    fn run(&self, img: ImageU8) -> Result<Tensor, JsError> {
        match &self.inner {
            Processor::Image(p) => {
                let out = p.preprocess(&img).map_err(js_err)?;
                let mut shape = vec![1u32];
                shape.extend(out.shape().iter().map(|&d| d as u32));
                Ok(Tensor { data: out.into_raw_vec_and_offset().0, shape, grid_thw: None })
            }
            Processor::Qwen2VL(p) => {
                let out = p.preprocess_batch(std::slice::from_ref(&img)).map_err(js_err)?;
                let shape = out.pixel_values.shape().iter().map(|&d| d as u32).collect();
                let grid = out.image_grid_thw.iter().map(|&v| v as i32).collect();
                Ok(Tensor { data: out.pixel_values.into_raw_vec_and_offset().0, shape, grid_thw: Some(grid) })
            }
            Processor::Whisper(_) => Err(JsError::new("this is an audio feature extractor; use extractAudio")),
        }
    }
}
