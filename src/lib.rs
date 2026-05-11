pub mod audio;
pub mod model;
pub mod phonemizer;
pub mod synthesis;

use candle_core::Device;

/// Returns the default device: Metal if the `metal` feature is enabled, otherwise CPU.
pub fn default_device() -> Device {
    #[cfg(feature = "metal")]
    {
        Device::new_metal(0).expect("Metal device not available")
    }
    #[cfg(not(feature = "metal"))]
    {
        Device::Cpu
    }
}
