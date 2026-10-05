//! CLIP ViT-B/32 image embeddings in pure Rust: hf-processors for the preprocessing
//! (bit-identical to transformers' `CLIPProcessor`), candle for the model, on CPU.
//!
//! ```text
//! cargo run --release -- <model_dir> <image_dir> <out_dir>
//! ```
//!
//! `model_dir` holds `model.safetensors` and `preprocessor_config.json` from
//! `openai/clip-vit-base-patch32` (a Hub repo id also works for the config, but the weights are
//! read from the directory). Every `.jpg` / `.jpeg` / `.png` in `image_dir` is embedded. The
//! example writes `files.txt`, `pixel_values.npy` (N, 3, 224, 224) and `image_embeds.npy`
//! (N, 512, `CLIPModel.get_image_features`) to `out_dir`; `scripts/real/compare_clip.py`
//! compares them with transformers.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip::{ClipConfig, ClipModel};
use hf_processors::{AutoImageProcessor, load_image, processor::to_candle};
use ndarray::{Array2, Axis};
use ndarray_npy::write_npy;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::time::Instant;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: {} <model_dir> <image_dir> <out_dir>", args[0]);
        std::process::exit(2);
    }
    let (model_dir, image_dir, out_dir) = (Path::new(&args[1]), Path::new(&args[2]), Path::new(&args[3]));
    std::fs::create_dir_all(out_dir)?;

    let mut files: Vec<PathBuf> = std::fs::read_dir(image_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().and_then(|e| e.to_str()).is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png"))
        })
        .collect();
    files.sort();

    // Preprocessing: the same preprocessor_config.json transformers reads; torchvision backend
    // (transformers v5's default CLIPImageProcessor).
    let t = Instant::now();
    let proc = AutoImageProcessor::from_pretrained(model_dir.to_str().ok_or("non-UTF-8 path")?)?;
    let images = files.par_iter().map(load_image).collect::<Result<Vec<_>, _>>()?;
    let pixel_values = proc.preprocess_batch(&images)?;
    println!("decoded + preprocessed {} images in {:.2} s: {:?}", files.len(), t.elapsed().as_secs_f64(), pixel_values.dim());

    // Model: CLIPModel.get_image_features = visual_projection(vision_model(x).pooler_output).
    let device = Device::Cpu;
    let weights = model_dir.join("model.safetensors");
    // SAFETY: the safetensors file is not modified while it is memory-mapped.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device)? };
    let model = ClipModel::new(vb, &ClipConfig::vit_base_patch32())?;
    let t = Instant::now();
    let mut chunks = Vec::new();
    for batch in pixel_values.axis_chunks_iter(Axis(0), 16) {
        let x = to_candle(&batch.to_owned(), &device)?;
        chunks.push(model.get_image_features(&x)?);
    }
    let embeds = Tensor::cat(&chunks, 0)?;
    let (n, d) = embeds.dims2()?;
    println!("embedded {n} images in {:.2} s: ({n}, {d})", t.elapsed().as_secs_f64());

    let embeds = Array2::from_shape_vec((n, d), embeds.flatten_all()?.to_vec1::<f32>()?)?;
    write_npy(out_dir.join("image_embeds.npy"), &embeds)?;
    write_npy(out_dir.join("pixel_values.npy"), &pixel_values)?;
    let names: Vec<String> = files.iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect();
    std::fs::write(out_dir.join("files.txt"), names.join("\n") + "\n")?;
    println!("wrote files.txt, pixel_values.npy, image_embeds.npy to {}", out_dir.display());
    Ok(())
}
