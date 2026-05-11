#![allow(dead_code)]

use candle_core::{DType, Result, Tensor};
use candle_nn::{Module, VarBuilder};

pub const KOKORO_SAMPLE_RATE: f32 = 24_000.0;
pub const KOKORO_UPSAMPLE_SCALE: usize = 300;
pub const KOKORO_HARMONIC_NUM: usize = 8;
pub const KOKORO_VOICED_THRESHOLD: f32 = 10.0;

pub struct SineGen {
    sampling_rate: f32,
    upsample_scale: usize,
    harmonic_num: usize,
    sine_amp: f32,
    noise_std: f32,
    voiced_threshold: f32,
}

impl Default for SineGen {
    fn default() -> Self {
        Self {
            sampling_rate: KOKORO_SAMPLE_RATE,
            upsample_scale: KOKORO_UPSAMPLE_SCALE,
            harmonic_num: KOKORO_HARMONIC_NUM,
            sine_amp: 0.1,
            noise_std: 0.003,
            voiced_threshold: KOKORO_VOICED_THRESHOLD,
        }
    }
}

impl SineGen {
    pub fn new(
        sampling_rate: f32,
        upsample_scale: usize,
        harmonic_num: usize,
        sine_amp: f32,
        noise_std: f32,
        voiced_threshold: f32,
    ) -> Self {
        Self {
            sampling_rate,
            upsample_scale,
            harmonic_num,
            sine_amp,
            noise_std,
            voiced_threshold,
        }
    }

    pub fn forward_with_controls(
        &self,
        f0: &Tensor,
        rand_ini: Option<&Tensor>,
        noise: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let shape = f0.dims();
        if shape.len() != 3 || shape[2] != 1 {
            return Err(candle_core::Error::Msg(format!(
                "SineGen expects [B, T, 1] f0, got {shape:?}"
            )));
        }
        let batch = shape[0];
        let time = shape[1];
        let dim = self.harmonic_num + 1;
        let device = f0.device();
        let f0 = f0.to_dtype(DType::F32)?;
        let uv = f0.gt(self.voiced_threshold as f64)?.to_dtype(DType::F32)?;

        let harmonic_scales = Tensor::from_vec(
            (1..=dim)
                .map(|h| h as f32 / self.sampling_rate)
                .collect::<Vec<_>>(),
            (1, 1, dim),
            device,
        )?;
        let rad = f0.broadcast_mul(&harmonic_scales)?;
        let rad = (rad.clone() - rad.floor()?)?;

        let rand_ini = match rand_ini {
            Some(rand_ini) => rand_ini.to_dtype(DType::F32)?,
            None => Tensor::zeros((batch, dim), DType::F32, device)?,
        };
        let first_frame_mask = Tensor::arange(0u32, time as u32, device)?
            .eq(0u32)?
            .to_dtype(DType::F32)?
            .reshape((1, time, 1))?;
        let rad = (rad
            + rand_ini
                .unsqueeze(1)?
                .broadcast_mul(&first_frame_mask)?
                .broadcast_as((batch, time, dim))?)?;

        let down = interpolate_linear_btd_tensor(&rad, 1.0 / self.upsample_scale as f64)?;
        let down_time = down.dim(1)?;
        // Cumsum needs f64 accumulation: f32 drifts ~1ulp/step which compounds
        // to ~0.02 rad over T_f0≈120 frames after the 2π·upsample_scale multiply,
        // producing audible artifacts (end-to-end audio diff 2.8e-2 with f32-only
        // vs 1.5e-6 with f64). candle 0.10's Metal backend has no f32↔f64
        // to_dtype, so do this step on CPU; `down` is [B, T_f0, harmonic+1] —
        // orders of magnitude smaller than the original full-time CPU stretch.
        let phase = down
            .to_device(&candle_core::Device::Cpu)?
            .to_dtype(DType::F64)?
            .cumsum(1)?
            .to_dtype(DType::F32)?
            .to_device(device)?
            .affine(
                2.0 * std::f64::consts::PI * self.upsample_scale as f64,
                0.0,
            )?;
        let phase = interpolate_linear_btd_tensor(&phase, self.upsample_scale as f64)?;
        let out_time = phase.dim(1)?;
        if down_time == 0 || out_time == 0 {
            return Err(candle_core::Error::Msg(
                "SineGen produced an empty phase tensor".to_string(),
            ));
        }

        let noise = match noise {
            Some(noise) => noise.to_dtype(DType::F32)?,
            None => Tensor::zeros((batch, out_time, dim), DType::F32, device)?,
        };
        let noise = if noise.dim(1)? == out_time {
            noise
        } else {
            noise.narrow(1, 0, out_time)?
        };

        let uv_out = uv.narrow(1, 0, out_time)?.broadcast_as((batch, out_time, dim))?;
        let inv_uv = uv_out.affine(-1.0, 1.0)?;
        let noise_amp = (uv_out.affine(self.noise_std as f64, 0.0)?
            + inv_uv.affine((self.sine_amp / 3.0) as f64, 0.0)?)?;
        let noise_out = noise_amp.broadcast_mul(&noise)?;
        let sine = (phase.sin()?.affine(self.sine_amp as f64, 0.0)?.broadcast_mul(&uv_out)?
            + &noise_out)?;
        Ok((sine, uv, noise_out))
    }

    pub fn forward(&self, f0: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        self.forward_with_controls(f0, None, None)
    }
}

pub struct SourceModuleHnNsf {
    sine_gen: SineGen,
    linear: candle_nn::Linear,
}

impl SourceModuleHnNsf {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            sine_gen: SineGen::default(),
            linear: candle_nn::linear(KOKORO_HARMONIC_NUM + 1, 1, vb.pp("l_linear"))?,
        })
    }

    pub fn forward_with_controls(
        &self,
        f0: &Tensor,
        rand_ini: Option<&Tensor>,
        noise: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (sine_waves, uv, sine_noise) =
            self.sine_gen.forward_with_controls(f0, rand_ini, noise)?;
        let sine_merge = self.linear.forward(&sine_waves)?.tanh()?;
        Ok((sine_merge, sine_noise, uv))
    }

    pub fn forward(&self, f0: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        self.forward_with_controls(f0, None, None)
    }
}

fn interpolate_linear_btd_tensor(values: &Tensor, scale_factor: f64) -> Result<Tensor> {
    let in_time = values.dim(1)?;
    let out_time = ((in_time as f64) * scale_factor).floor() as usize;
    let device = values.device();
    if out_time == 0 {
        return values.narrow(1, 0, 0);
    }
    let mut idx0 = Vec::with_capacity(out_time);
    let mut idx1 = Vec::with_capacity(out_time);
    let mut frac = Vec::with_capacity(out_time);
    for t_out in 0..out_time {
        let in_pos = ((t_out as f64 + 0.5) / scale_factor) - 0.5;
        let in_pos = in_pos.clamp(0.0, (in_time - 1) as f64);
        let t0 = in_pos.floor() as u32;
        let t1 = (t0 as usize + 1).min(in_time - 1) as u32;
        idx0.push(t0);
        idx1.push(t1);
        frac.push((in_pos - t0 as f64) as f32);
    }
    let idx0 = Tensor::from_vec(idx0, out_time, device)?;
    let idx1 = Tensor::from_vec(idx1, out_time, device)?;
    let frac = Tensor::from_vec(frac, (1, out_time, 1), device)?;
    let v0 = values.index_select(&idx0, 1)?;
    let v1 = values.index_select(&idx1, 1)?;
    (v0.broadcast_mul(&frac.affine(-1.0, 1.0)?)? + v1.broadcast_mul(&frac)?)?
        .contiguous()
}
