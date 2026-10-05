//! Throughput benchmark: preprocess a batch of already-decoded images.
//!
//!     cargo run --release --example bench --features rayon -- [threads]
//!
//! Mirrors `golden/bench.py` (same images, same configs).

use hf_processors::{Backend, ImageProcessor, ImageU8, PreprocessorConfig, ProcessorKind, load_image};
use std::path::Path;
use std::time::Instant;

fn golden() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/golden"))
}

fn batch(names: &[&str], n: usize) -> Vec<ImageU8> {
    let imgs: Vec<ImageU8> = names.iter().map(|n| load_image(golden().join("images").join(n)).unwrap()).collect();
    (0..n).map(|i| imgs[i % imgs.len()].clone()).collect()
}

fn time_it(proc: &ImageProcessor, images: &[ImageU8], iters: usize) -> f64 {
    let _ = proc.preprocess_batch(images).unwrap(); // warm-up
    let mut times: Vec<f64> = (0..iters)
        .map(|_| {
            let t = Instant::now();
            let out = proc.preprocess_batch(images).unwrap();
            std::hint::black_box(out);
            t.elapsed().as_secs_f64()
        })
        .collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

fn main() {
    let threads: usize = std::env::args().nth(1).map(|s| s.parse().unwrap()).unwrap_or(1);
    #[cfg(feature = "rayon")]
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();
    #[cfg(not(feature = "rayon"))]
    assert_eq!(threads, 1, "build with --features rayon for multi-threaded batches");

    let typical =
        batch(&["photo_640x480.png", "coffee.png", "astronaut.png", "hifreq_333x517.png", "photo_500x375.jpg"], 64);
    let large = batch(&["huge_3000x2000.png"], 8);
    let configs = [
        ("clip-vit-base-patch32", "openai_clip-vit-base-patch32.json", ProcessorKind::Clip),
        ("vit-base-patch16-224", "google_vit-base-patch16-224.json", ProcessorKind::Vit),
        ("siglip-so400m-patch14-384", "google_siglip-so400m-patch14-384.json", ProcessorKind::Siglip),
    ];
    println!("threads={threads}");
    println!("{:<28} {:<12} {:<26} {:>10} {:>12}", "config", "backend", "batch", "ms/batch", "images/s");
    for (name, file, kind) in configs {
        let cfg = PreprocessorConfig::from_file(golden().join("configs").join(file)).unwrap();
        for backend in [Backend::Torchvision, Backend::Pil] {
            let proc = ImageProcessor::from_config_as(&cfg, kind).unwrap().with_backend(backend);
            for (label, imgs, iters) in [("64 x ~0.3MP (typical)", &typical, 15), ("8 x 6MP (3000x2000)", &large, 5)] {
                let t = time_it(&proc, imgs, iters);
                println!(
                    "{:<28} {:<12} {:<26} {:>10.1} {:>12.0}",
                    name,
                    format!("{backend:?}"),
                    label,
                    t * 1e3,
                    imgs.len() as f64 / t
                );
            }
        }
    }
}
