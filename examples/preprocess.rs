//! Preprocess one image with a Hub/local processor config and print a summary.
//!
//!     cargo run --release --example preprocess -- openai/clip-vit-base-patch32 golden/images/coffee.png [pil|torchvision]

use hf_processors::{AutoImageProcessor, Backend};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let repo = args.next().ok_or("usage: preprocess <repo_or_path> <image> [pil|torchvision]")?;
    let image = args.next().ok_or("missing image path")?;
    let backend = match args.next().as_deref() {
        Some("pil") => Backend::Pil,
        _ => Backend::Torchvision,
    };
    let proc = AutoImageProcessor::from_pretrained(&repo)?.with_backend(backend);
    println!(
        "{:?} backend={:?} size={:?} crop={:?} resample={:?}",
        proc.kind, proc.backend, proc.size, proc.crop_size, proc.resample
    );
    let pv = proc.preprocess_path(&image)?;
    let sum: f64 = pv.iter().map(|&v| v as f64).sum();
    println!("pixel_values shape={:?} sum={sum:.6} first={:?}", pv.shape(), pv.iter().take(4).collect::<Vec<_>>());
    Ok(())
}
