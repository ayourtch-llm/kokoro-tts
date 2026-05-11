pub mod audio;
pub mod model;
pub mod phonemizer;
pub mod synthesis;

use candle_core::Device;

/// Returns the default device: Metal if available with the `metal` feature, otherwise CPU.
pub fn default_device() -> Device {
    #[cfg(feature = "metal")]
    {
        match Device::new_metal(0) {
            Ok(d) => return d,
            Err(e) => tracing::warn!("Metal unavailable, falling back to CPU: {e}"),
        }
    }
    Device::Cpu
}
