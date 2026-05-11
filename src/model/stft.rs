#![allow(dead_code)]

use candle_core::{Result, Tensor, D};

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

    pub fn inverse(
        &self,
        magnitude: &Tensor,
        phase: &Tensor,
        length: Option<usize>,
    ) -> Result<Tensor> {
        let real_part = (magnitude * &phase.cos()?)?;
        let imag_part = (magnitude * &phase.sin()?)?;
        let real_rec =
            real_part.conv_transpose1d(&self.weight_backward_real, 0, 0, self.hop_length, 1, 1)?;
        let imag_rec =
            imag_part.conv_transpose1d(&self.weight_backward_imag, 0, 0, self.hop_length, 1, 1)?;
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
    let zero = x.zeros_like()?;
    let one = x.ones_like()?;
    let x_zero = x.eq(0f64)?;
    let safe_abs_x = x_zero.where_cond(&one, &x.abs()?)?;
    let ratio = y.abs()?.broadcast_div(&safe_abs_x)?;
    let theta = atan_nonnegative(&ratio)?;

    let signed = y.lt(0f64)?.where_cond(&theta.neg()?, &theta)?;
    let x_negative = x.lt(0f64)?;
    let x_negative_value = y.lt(0f64)?.where_cond(
        &theta.affine(1.0, -std::f64::consts::PI)?,
        &theta.affine(-1.0, std::f64::consts::PI)?,
    )?;
    let nonzero_x = x_negative.where_cond(&x_negative_value, &signed)?;

    let half_pi = zero.affine(0.0, std::f64::consts::FRAC_PI_2)?;
    let neg_half_pi = zero.affine(0.0, -std::f64::consts::FRAC_PI_2)?;
    let zero_x_value = y
        .gt(0f64)?
        .where_cond(&half_pi, &y.lt(0f64)?.where_cond(&neg_half_pi, &zero)?)?;
    x_zero.where_cond(&zero_x_value, &nonzero_x)
}

fn atan_nonnegative(z: &Tensor) -> Result<Tensor> {
    let tan_pi_8 = std::f64::consts::SQRT_2 - 1.0;
    let tan_3pi_8 = std::f64::consts::SQRT_2 + 1.0;

    let low = atan_poly(z)?;
    let mid_arg = (z - 1.0)?.broadcast_div(&(z + 1.0)?)?;
    let mid = atan_poly(&mid_arg)?.affine(1.0, std::f64::consts::FRAC_PI_4)?;
    let high = atan_poly(&z.recip()?)?.affine(-1.0, std::f64::consts::FRAC_PI_2)?;

    let low_or_mid = z.gt(tan_pi_8)?.where_cond(&mid, &low)?;
    z.gt(tan_3pi_8)?.where_cond(&high, &low_or_mid)
}

fn atan_poly(z: &Tensor) -> Result<Tensor> {
    let z2 = z.sqr()?;
    let mut acc = z2.affine(0.0, 1.0 / 29.0)?;
    for k in (0..14).rev() {
        let coeff = if k % 2 == 0 { 1.0 } else { -1.0 } / (2 * k + 1) as f64;
        acc = z2.broadcast_mul(&acc)?.affine(1.0, coeff)?;
    }
    z.broadcast_mul(&acc)
}
