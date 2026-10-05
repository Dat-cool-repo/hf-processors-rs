//! Image primitives: buffers, Pillow- and PyTorch-exact resizing, crops, normalization.

pub mod buffer;
mod kernels;
pub mod ops;
pub mod pil_resize;
pub mod torch_resize;

pub use buffer::ImageU8;
#[cfg(feature = "decode")]
pub use buffer::{decode_image, load_image};
pub use pil_resize::{PassOrder, Resample};
pub use torch_resize::TorchInterp;
