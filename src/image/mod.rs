//! Image primitives: buffers, Pillow- and PyTorch-exact resizing, crops, normalization.

pub mod buffer;
mod kernels;
pub mod ops;
pub mod pil_resize;
pub mod torch_resize;

pub use buffer::ImageU8;
#[cfg(feature = "decode")]
pub use buffer::{decode_image, load_image};
#[cfg(feature = "decode")]
#[doc(hidden)]
pub use buffer::decode_image_pure_rust;
/// Force the portable (non-SIMD) resize kernels, process-wide. Both are exact; used to fuzz
/// one against the other.
#[doc(hidden)]
pub use kernels::set_force_scalar as __set_force_scalar_kernels;
pub use pil_resize::{PassOrder, Resample};
pub use torch_resize::TorchInterp;
