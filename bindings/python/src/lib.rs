//! Native part of the `hf_processors_rs` Python package. The user-facing API (PIL handling,
//! `AutoImageProcessor`, ...) lives in `python/hf_processors_rs/__init__.py`.
//!
//! Inputs are copied into Rust buffers while holding the GIL; decoding and preprocessing then
//! run with the GIL released, in parallel over the batch (rayon).

use hf_processors::{
    AutoProcessor, Backend, Error, ImageU8, Padding, PreprocessorConfig, Processor, WhisperOptions, decode_image,
    load_image,
};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyOSError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

fn to_py_err(e: Error) -> PyErr {
    match e {
        Error::Io(e) => PyOSError::new_err(e.to_string()),
        Error::Hub(m) => PyOSError::new_err(m),
        Error::UnsupportedProcessor(m) => PyValueError::new_err(format!("unsupported processor: {m}")),
        other => PyValueError::new_err(other.to_string()),
    }
}

fn parse_backend(s: Option<&str>) -> PyResult<Backend> {
    match s.unwrap_or("torchvision") {
        "torchvision" | "fast" => Ok(Backend::Torchvision),
        "pil" | "slow" => Ok(Backend::Pil),
        other => Err(PyValueError::new_err(format!("backend must be 'torchvision' or 'pil', got {other:?}"))),
    }
}

fn backend_name(b: Backend) -> &'static str {
    match b {
        Backend::Torchvision => "torchvision",
        Backend::Pil => "pil",
    }
}

/// Thread pools by size (built once), for `num_threads=`.
fn pool(n: usize) -> PyResult<Arc<rayon::ThreadPool>> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();
    let mut map = POOLS.get_or_init(Default::default).lock().unwrap();
    if let Some(p) = map.get(&n) {
        return Ok(p.clone());
    }
    let p = Arc::new(
        rayon::ThreadPoolBuilder::new().num_threads(n).build().map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    );
    map.insert(n, p.clone());
    Ok(p)
}

fn run_in_pool<T: Send>(num_threads: Option<usize>, f: impl FnOnce() -> T + Send) -> PyResult<T> {
    match num_threads {
        None | Some(0) => Ok(f()),
        Some(n) => Ok(pool(n)?.install(f)),
    }
}

/// A C-contiguous uint8 numpy buffer, read after the GIL is released. The array object is kept
/// alive (and read-only borrowed through rust-numpy) by the caller for the whole call.
struct RawPixels {
    ptr: *const u8,
    len: usize,
    h: usize,
    w: usize,
    c: usize,
    chw: bool,
}

// SAFETY: the pointed-to buffer outlives the GIL-released section (the `PyReadonlyArray` that
// owns the borrow is held by the calling frame), and it is only read.
unsafe impl Send for RawPixels {}

/// One image as received from Python.
enum Input {
    Pixels(ImageU8),
    Raw(RawPixels),
    Path(PathBuf),
    Encoded(Vec<u8>),
}

impl Input {
    fn load(self) -> hf_processors::Result<ImageU8> {
        match self {
            Input::Pixels(img) => Ok(img),
            Input::Raw(r) => {
                // SAFETY: see `RawPixels`.
                let src = unsafe { std::slice::from_raw_parts(r.ptr, r.len) };
                let data = if r.chw {
                    let plane = r.h * r.w;
                    let mut d = vec![0u8; r.len];
                    for (i, px) in d.chunks_exact_mut(r.c).enumerate() {
                        for (ch, v) in px.iter_mut().enumerate() {
                            *v = src[ch * plane + i];
                        }
                    }
                    d
                } else {
                    src.to_vec()
                };
                ImageU8::new(r.w, r.h, r.c, data)
            }
            Input::Path(p) => load_image(p),
            Input::Encoded(b) => decode_image(&b),
        }
    }
}

/// `(h, w, c, channels_first)` of a uint8 array. `layout`: "channels_last", "channels_first" or
/// None (infer like transformers: a leading dimension of 1 or 3 means channels-first).
fn array_layout(shape: &[usize], layout: Option<&str>) -> PyResult<(usize, usize, usize, bool)> {
    let dims = match (shape.len(), layout) {
        (2, _) => (shape[0], shape[1], 1, false),
        (3, Some("channels_first")) => (shape[1], shape[2], shape[0], true),
        (3, Some("channels_last")) => (shape[0], shape[1], shape[2], false),
        (3, None) if matches!(shape[0], 1 | 3) => (shape[1], shape[2], shape[0], true),
        (3, None) if (1..=4).contains(&shape[2]) => (shape[0], shape[1], shape[2], false),
        _ => {
            return Err(PyValueError::new_err(format!(
                "unsupported image array shape {shape:?}: expected (H, W), (H, W, C) or (C, H, W) with C in 1..=4"
            )));
        }
    };
    if !(1..=4).contains(&dims.2) || dims.0 == 0 || dims.1 == 0 {
        return Err(PyValueError::new_err(format!("unsupported image array shape {shape:?}")));
    }
    Ok(dims)
}

/// A contiguous array becomes a zero-copy `Input::Raw` (copied later, in parallel, without
/// the GIL); anything else is copied now.
fn array_input(a: &PyReadonlyArrayDyn<u8>, layout: Option<&str>) -> PyResult<Input> {
    let (h, w, c, chw) = array_layout(a.shape(), layout)?;
    // `as_slice` also accepts Fortran-order arrays: require C order explicitly.
    if a.is_c_contiguous()
        && let Ok(s) = a.as_slice()
    {
        return Ok(Input::Raw(RawPixels { ptr: s.as_ptr(), len: s.len(), h, w, c, chw }));
    }
    Ok(Input::Pixels(array_to_image(a, layout)?))
}

/// uint8 array -> HWC image (copying).
fn array_to_image(a: &PyReadonlyArrayDyn<u8>, layout: Option<&str>) -> PyResult<ImageU8> {
    let shape = a.shape().to_vec();
    let view = a.as_array();
    let (h, w, c, chw) = array_layout(&shape, layout)?;
    let data: Vec<u8> = if shape.len() == 2 {
        view.iter().copied().collect()
    } else if chw {
        let v = view.into_dimensionality::<numpy::ndarray::Ix3>().unwrap();
        let v = v.permuted_axes([1, 2, 0]);
        v.iter().copied().collect()
    } else if let Some(s) = view.as_slice() {
        s.to_vec()
    } else {
        view.iter().copied().collect()
    };
    ImageU8::new(w, h, c, data).map_err(to_py_err)
}

/// Python inputs -> `Input`s. Borrowed numpy arrays are pushed to `keep`, which the caller must
/// hold until the GIL-released work is done.
fn extract_inputs<'py>(
    images: &Bound<'py, PyList>,
    layout: Option<&str>,
    keep: &mut Vec<PyReadonlyArrayDyn<'py, u8>>,
) -> PyResult<Vec<Input>> {
    let mut out = Vec::with_capacity(images.len());
    for item in images.iter() {
        if let Ok(b) = item.cast::<PyBytes>() {
            out.push(Input::Encoded(b.as_bytes().to_vec()));
        } else if let Ok((a, l)) = item.extract::<(PyReadonlyArrayDyn<u8>, String)>() {
            // (array, layout): explicit layout, e.g. arrays converted from PIL images (always HWC).
            out.push(array_input(&a, Some(l.as_str()))?);
            keep.push(a);
        } else if let Ok(a) = item.extract::<PyReadonlyArrayDyn<u8>>() {
            out.push(array_input(&a, layout)?);
            keep.push(a);
        } else if let Ok(p) = item.extract::<PathBuf>() {
            out.push(Input::Path(p));
        } else {
            return Err(PyTypeError::new_err(format!(
                "unsupported image input of type {}: pass a PIL image, a uint8 numpy array, a path or encoded bytes",
                item.get_type().name()?
            )));
        }
    }
    if out.is_empty() {
        return Err(PyValueError::new_err("no images given"));
    }
    Ok(out)
}

fn load_all(inputs: Vec<Input>) -> hf_processors::Result<Vec<ImageU8>> {
    inputs.into_par_iter().map(Input::load).collect()
}

fn vec_to_numpy<'py, T: numpy::Element>(py: Python<'py>, v: Vec<T>, shape: &[usize]) -> PyResult<Bound<'py, PyAny>> {
    let arr = PyArray1::from_vec(py, v);
    Ok(arr.reshape(shape)?.into_any())
}

/// A loaded processor (any kind). Wrapped by the Python classes in `__init__.py`.
#[pyclass(module = "hf_processors_rs._hf_processors_rs", name = "NativeProcessor")]
struct NativeProcessor {
    inner: Processor,
}

impl NativeProcessor {
    fn with_backend(mut inner: Processor, backend: Backend) -> Self {
        match &mut inner {
            Processor::Image(p) => p.backend = backend,
            Processor::Qwen2VL(p) => p.backend = backend,
            Processor::Whisper(_) => {}
        }
        NativeProcessor { inner }
    }
}

#[pymethods]
impl NativeProcessor {
    /// Load from a Hub repo id, a directory or a `preprocessor_config.json` path.
    #[staticmethod]
    #[pyo3(signature = (repo_or_path, backend=None))]
    fn from_pretrained(py: Python<'_>, repo_or_path: PathBuf, backend: Option<&str>) -> PyResult<Self> {
        let backend = parse_backend(backend)?;
        let s = repo_or_path.to_string_lossy().into_owned();
        let inner = py.detach(|| AutoProcessor::from_pretrained(&s)).map_err(to_py_err)?;
        Ok(Self::with_backend(inner, backend))
    }

    /// Build from the JSON text of a `preprocessor_config.json`. `processor_type` overrides
    /// the class named in the config (needed when it has none).
    #[staticmethod]
    #[pyo3(signature = (json, backend=None, processor_type=None))]
    fn from_json(json: &str, backend: Option<&str>, processor_type: Option<&str>) -> PyResult<Self> {
        let backend = parse_backend(backend)?;
        let cfg = PreprocessorConfig::from_json_str(json).map_err(to_py_err)?;
        let inner = match processor_type {
            Some(t) => {
                let mut c = cfg.clone();
                c.image_processor_type = Some(t.to_string());
                AutoProcessor::from_config_as(&cfg, &c.processor_type().unwrap())
            }
            None => AutoProcessor::from_config(&cfg),
        }
        .map_err(to_py_err)?;
        Ok(Self::with_backend(inner, backend))
    }

    /// `"image"`, `"qwen2_vl"` or `"whisper"`.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            Processor::Image(_) => "image",
            Processor::Qwen2VL(_) => "qwen2_vl",
            Processor::Whisper(_) => "whisper",
        }
    }

    /// The transformers class reproduced (torchvision-backend name).
    #[getter]
    fn class_name(&self) -> &'static str {
        self.inner.class_name()
    }

    #[getter]
    fn backend(&self) -> Option<&'static str> {
        match &self.inner {
            Processor::Image(p) => Some(backend_name(p.backend)),
            Processor::Qwen2VL(p) => Some(backend_name(p.backend)),
            Processor::Whisper(_) => None,
        }
    }

    #[setter]
    fn set_backend(&mut self, backend: &str) -> PyResult<()> {
        let b = parse_backend(Some(backend))?;
        match &mut self.inner {
            Processor::Image(p) => p.backend = b,
            Processor::Qwen2VL(p) => p.backend = b,
            Processor::Whisper(_) => return Err(PyValueError::new_err("audio feature extractors have no backend")),
        }
        Ok(())
    }

    /// Resolved settings, for inspection.
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        match &self.inner {
            Processor::Image(p) => {
                d.set_item("do_convert_rgb", p.do_convert_rgb)?;
                d.set_item("do_resize", p.do_resize)?;
                d.set_item("size", format!("{:?}", p.size))?;
                d.set_item("resample", p.resample as i64)?;
                d.set_item("do_center_crop", p.do_center_crop)?;
                d.set_item("crop_size", p.crop_size)?;
                d.set_item("do_rescale", p.do_rescale)?;
                d.set_item("rescale_factor", p.rescale_factor)?;
                d.set_item("do_normalize", p.do_normalize)?;
                d.set_item("image_mean", p.image_mean.clone())?;
                d.set_item("image_std", p.image_std.clone())?;
            }
            Processor::Qwen2VL(p) => {
                d.set_item("do_convert_rgb", p.do_convert_rgb)?;
                d.set_item("do_resize", p.do_resize)?;
                d.set_item("min_pixels", p.min_pixels)?;
                d.set_item("max_pixels", p.max_pixels)?;
                d.set_item("resample", p.resample as i64)?;
                d.set_item("do_rescale", p.do_rescale)?;
                d.set_item("rescale_factor", p.rescale_factor)?;
                d.set_item("do_normalize", p.do_normalize)?;
                d.set_item("image_mean", p.image_mean.clone())?;
                d.set_item("image_std", p.image_std.clone())?;
                d.set_item("patch_size", p.patch_size)?;
                d.set_item("temporal_patch_size", p.temporal_patch_size)?;
                d.set_item("merge_size", p.merge_size)?;
            }
            Processor::Whisper(p) => {
                d.set_item("feature_size", p.feature_size)?;
                d.set_item("sampling_rate", p.sampling_rate)?;
                d.set_item("hop_length", p.hop_length)?;
                d.set_item("chunk_length", p.chunk_length)?;
                d.set_item("n_fft", p.n_fft)?;
                d.set_item("n_samples", p.n_samples)?;
                d.set_item("padding_value", p.padding_value)?;
                d.set_item("dither", p.dither)?;
                d.set_item("return_attention_mask", p.return_attention_mask)?;
            }
        }
        Ok(d)
    }

    /// Fixed-size processors: `pixel_values` `(N, C, H, W)` float32.
    #[pyo3(signature = (images, num_threads=None, input_data_format=None))]
    fn preprocess_images<'py>(
        &self,
        py: Python<'py>,
        images: &Bound<'py, PyList>,
        num_threads: Option<usize>,
        input_data_format: Option<&str>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let Processor::Image(p) = &self.inner else {
            return Err(PyTypeError::new_err(format!("{} is not a fixed-size image processor", self.class_name())));
        };
        let mut keep = Vec::new();
        let inputs = extract_inputs(images, input_data_format, &mut keep)?;
        let p = p.clone();
        let out = py
            .detach(|| run_in_pool(num_threads, || load_all(inputs).and_then(|imgs| p.preprocess_batch(&imgs))))?
            .map_err(to_py_err)?;
        drop(keep); // input borrows end only after the GIL-free section
        let shape = out.shape().to_vec();
        let (v, _) = out.into_raw_vec_and_offset();
        vec_to_numpy(py, v, &shape)
    }

    /// Qwen2-VL: `(pixel_values (P, D) float32, image_grid_thw (N, 3) int64)`.
    #[pyo3(signature = (images, num_threads=None, input_data_format=None))]
    fn preprocess_qwen<'py>(
        &self,
        py: Python<'py>,
        images: &Bound<'py, PyList>,
        num_threads: Option<usize>,
        input_data_format: Option<&str>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let Processor::Qwen2VL(p) = &self.inner else {
            return Err(PyTypeError::new_err(format!("{} is not a Qwen2-VL processor", self.class_name())));
        };
        let mut keep = Vec::new();
        let inputs = extract_inputs(images, input_data_format, &mut keep)?;
        let p = p.clone();
        let out = py
            .detach(|| run_in_pool(num_threads, || load_all(inputs).and_then(|imgs| p.preprocess_batch(&imgs))))?
            .map_err(to_py_err)?;
        drop(keep);
        let pv_shape = out.pixel_values.shape().to_vec();
        let g_shape = out.image_grid_thw.shape().to_vec();
        let (pv, _) = out.pixel_values.into_raw_vec_and_offset();
        let (g, _) = out.image_grid_thw.into_raw_vec_and_offset();
        Ok((vec_to_numpy(py, pv, &pv_shape)?, vec_to_numpy(py, g, &g_shape)?))
    }

    /// Whisper: `(input_features (N, n_mels, frames) float32, attention_mask int32 or None)`.
    #[pyo3(signature = (clips, padding="max_length", max_length=None, truncation=true, pad_to_multiple_of=None,
                        return_attention_mask=None, do_normalize=None, dither_seed=0, noise=None))]
    #[allow(clippy::too_many_arguments)]
    fn extract_audio<'py>(
        &self,
        py: Python<'py>,
        clips: Vec<PyReadonlyArray1<'py, f32>>,
        padding: &str,
        max_length: Option<usize>,
        truncation: bool,
        pad_to_multiple_of: Option<usize>,
        return_attention_mask: Option<bool>,
        do_normalize: Option<bool>,
        dither_seed: u64,
        noise: Option<PyReadonlyArray1<'py, f32>>,
    ) -> PyResult<(Bound<'py, PyAny>, Option<Bound<'py, PyAny>>)> {
        let Processor::Whisper(fe) = &self.inner else {
            return Err(PyTypeError::new_err(format!("{} is not an audio feature extractor", self.class_name())));
        };
        let padding = match padding {
            "max_length" => Padding::MaxLength,
            "longest" => Padding::Longest,
            "do_not_pad" => Padding::DoNotPad,
            other => {
                return Err(PyValueError::new_err(format!(
                    "padding must be 'max_length', 'longest' or 'do_not_pad', got {other:?}"
                )));
            }
        };
        let opts = WhisperOptions {
            padding,
            max_length,
            truncation,
            pad_to_multiple_of,
            return_attention_mask,
            do_normalize,
            dither_seed,
        };
        let owned: Vec<Vec<f32>> = clips.iter().map(|c| c.as_array().iter().copied().collect()).collect();
        let noise: Option<Vec<f32>> = noise.map(|n| n.as_array().iter().copied().collect());
        let fe = fe.clone();
        let out = py
            .detach(|| {
                let refs: Vec<&[f32]> = owned.iter().map(|v| v.as_slice()).collect();
                fe.call_with_noise(&refs, &opts, noise.as_deref())
            })
            .map_err(to_py_err)?;
        let shape = out.input_features.shape().to_vec();
        let (v, _) = out.input_features.into_raw_vec_and_offset();
        let feats = vec_to_numpy(py, v, &shape)?;
        let mask = match out.attention_mask {
            Some(m) => {
                let s = m.shape().to_vec();
                let (v, _) = m.into_raw_vec_and_offset();
                Some(vec_to_numpy(py, v, &s)?)
            }
            None => None,
        };
        Ok((feats, mask))
    }

    fn __repr__(&self) -> String {
        match self.backend() {
            Some(b) => format!("NativeProcessor({}, backend={b})", self.class_name()),
            None => format!("NativeProcessor({})", self.class_name()),
        }
    }
}

/// Decode an image file like `PIL.Image.open` (no EXIF rotation) into a uint8 HWC array.
#[pyfunction]
fn load_image_array<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyAny>> {
    let img = py.detach(|| load_image(&path)).map_err(to_py_err)?;
    let shape = [img.height, img.width, img.channels];
    vec_to_numpy(py, img.data, &shape)
}

#[pymodule]
fn _hf_processors_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeProcessor>()?;
    m.add_function(wrap_pyfunction!(load_image_array, m)?)?;
    m.add("PIL_JPEG", cfg!(feature = "pil-jpeg"))?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
