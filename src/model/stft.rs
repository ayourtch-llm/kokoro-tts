#![allow(dead_code)]

use candle_core::{DType, Result, Tensor, D};

pub struct CustomStft {
    n_fft: usize,
    hop_length: usize,
    center: bool,
    weight_forward_real: Tensor,
    weight_forward_imag: Tensor,
    weight_backward_real: Tensor,
    weight_backward_imag: Tensor,
}

impl CustomStft {
    pub fn new(n_fft: usize, hop_length: usize, device: &candle_core::Device) -> Result<Self> {
        let freq_bins = n_fft / 2 + 1;
        let window = hann_window_periodic(n_fft);
        let mut forward_real = Vec::with_capacity(freq_bins * n_fft);
        let mut forward_imag = Vec::with_capacity(freq_bins * n_fft);
        let mut backward_real = Vec::with_capacity(freq_bins * n_fft);
        let mut backward_imag = Vec::with_capacity(freq_bins * n_fft);
        let inv_scale = 1.0f32 / n_fft as f32;

        for k in 0..freq_bins {
            for (n, &w) in window.iter().enumerate() {
                let angle = 2.0 * std::f32::consts::PI * (k * n) as f32 / n_fft as f32;
                forward_real.push(angle.cos() * w);
                forward_imag.push(-angle.sin() * w);
                backward_real.push(angle.cos() * w * inv_scale);
                backward_imag.push(angle.sin() * w * inv_scale);
            }
        }

        Ok(Self {
            n_fft,
            hop_length,
            center: true,
            weight_forward_real: Tensor::from_vec(forward_real, (freq_bins, 1, n_fft), device)?,
            weight_forward_imag: Tensor::from_vec(forward_imag, (freq_bins, 1, n_fft), device)?,
            weight_backward_real: Tensor::from_vec(backward_real, (freq_bins, 1, n_fft), device)?,
            weight_backward_imag: Tensor::from_vec(backward_imag, (freq_bins, 1, n_fft), device)?,
        })
    }

    pub fn transform(&self, waveform: &Tensor) -> Result<(Tensor, Tensor)> {
        let waveform = if self.center {
            waveform.pad_with_same(D::Minus1, self.n_fft / 2, self.n_fft / 2)?
        } else {
            waveform.clone()
        };
        let x = waveform.unsqueeze(1)?;
        let real = x.conv1d(&self.weight_forward_real, 0, self.hop_length, 1, 1)?;
        let imag = x.conv1d(&self.weight_forward_imag, 0, self.hop_length, 1, 1)?;
        // For real-valued input, the DFT's imaginary part at DC (bin 0) and at
        // Nyquist (bin n_fft/2) is mathematically zero. Float-precision noise
        // from the conv1d sum can leave them at ±tiny with platform-dependent
        // sign — torch's BLAS path and candle's path don't agree on the sign.
        // atan2 then returns ±π non-deterministically. Zeroing those bins is
        // mathematically correct for real input and removes the divergence.
        let imag = zero_dc_and_nyquist_imag(&imag, self.n_fft)?;
        let magnitude = ((real.sqr()? + imag.sqr()?)? + 1e-14f64)?.sqrt()?;
        let phase = atan2_tensor(&imag, &real)?;
        Ok((magnitude, phase))
    }

    pub fn inverse(&self, magnitude: &Tensor, phase: &Tensor, length: Option<usize>) -> Result<Tensor> {
        let real_part = (magnitude * &phase.cos()?)?;
        let imag_part = (magnitude * &phase.sin()?)?;
        let real_rec = real_part.conv_transpose1d(
            &self.weight_backward_real,
            0,
            0,
            self.hop_length,
            1,
            1,
        )?;
        let imag_rec = imag_part.conv_transpose1d(
            &self.weight_backward_imag,
            0,
            0,
            self.hop_length,
            1,
            1,
        )?;
        let mut waveform = (real_rec - imag_rec)?;
        if self.center {
            let pad = self.n_fft / 2;
            let len = waveform.dim(D::Minus1)? - 2 * pad;
            waveform = waveform.narrow(D::Minus1, pad, len)?;
        }
        if let Some(length) = length {
            waveform = waveform.narrow(D::Minus1, 0, length.min(waveform.dim(D::Minus1)?))?;
        }
        waveform.squeeze(1)
    }

    pub fn forward(&self, waveform: &Tensor) -> Result<Tensor> {
        let length = waveform.dim(D::Minus1)?;
        let (magnitude, phase) = self.transform(waveform)?;
        self.inverse(&magnitude, &phase, Some(length))
    }
}

fn hann_window_periodic(n_fft: usize) -> Vec<f32> {
    (0..n_fft)
        .map(|n| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / n_fft as f32).cos())
        .collect()
}

fn zero_dc_and_nyquist_imag(imag: &Tensor, n_fft: usize) -> Result<Tensor> {
    let freq_bins = n_fft / 2 + 1;
    if freq_bins < 2 {
        return Ok(imag.clone());
    }
    let batch = imag.dim(0)?;
    let time = imag.dim(2)?;
    let zeros_row = Tensor::zeros((batch, 1, time), imag.dtype(), imag.device())?;
    let middle = imag.narrow(1, 1, freq_bins - 2)?;
    Tensor::cat(&[&zeros_row, &middle, &zeros_row], 1)
}

fn atan2_tensor(y: &Tensor, x: &Tensor) -> Result<Tensor> {
    let y = y.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let x_vec = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let phase = y
        .iter()
        .zip(x_vec.iter())
        .map(|(&yy, &xx)| yy.atan2(xx))
        .collect::<Vec<_>>();
    Tensor::from_vec(phase, x.shape(), x.device())
}
