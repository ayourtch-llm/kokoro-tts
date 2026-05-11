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
        let f0_data = f0.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;

        let rand_ini = match rand_ini {
            Some(rand_ini) => rand_ini
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            None => vec![0.0; batch * dim],
        };
        let noise = match noise {
            Some(noise) => noise
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            None => vec![0.0; batch * time * dim],
        };

        let mut rad_values = vec![0.0f32; batch * time * dim];
        let mut uv = vec![0.0f32; batch * time];
        for b in 0..batch {
            for t in 0..time {
                let f0_value = f0_data[b * time + t];
                let voiced = if f0_value > self.voiced_threshold {
                    1.0
                } else {
                    0.0
                };
                uv[b * time + t] = voiced;
                for h in 0..dim {
                    let overtone = (h + 1) as f32;
                    let mut rad = (f0_value * overtone / self.sampling_rate).fract();
                    if t == 0 {
                        rad += rand_ini[b * dim + h];
                    }
                    rad_values[(b * time + t) * dim + h] = rad;
                }
            }
        }

        let down = interpolate_linear_btd(
            &rad_values,
            batch,
            time,
            dim,
            1.0 / self.upsample_scale as f64,
        );
        let down_time = down.len() / (batch * dim);
        let mut phase = vec![0.0f32; batch * down_time * dim];
        let scale = (2.0 * std::f64::consts::PI) as f32;
        for b in 0..batch {
            for h in 0..dim {
                let mut acc = 0.0f64;
                for t in 0..down_time {
                    acc += f64::from(down[(b * down_time + t) * dim + h]);
                    phase[(b * down_time + t) * dim + h] = (acc as f32) * scale;
                }
            }
        }
        for value in &mut phase {
            *value *= self.upsample_scale as f32;
        }
        let phase =
            interpolate_linear_btd(&phase, batch, down_time, dim, self.upsample_scale as f64);
        let out_time = phase.len() / (batch * dim);
        let mut sine_waves = vec![0.0f32; batch * out_time * dim];
        let mut noise_out = vec![0.0f32; batch * out_time * dim];
        for b in 0..batch {
            for t in 0..out_time {
                let uv_value = uv[b * time + t.min(time - 1)];
                let noise_amp = uv_value * self.noise_std + (1.0 - uv_value) * self.sine_amp / 3.0;
                for h in 0..dim {
                    let idx = (b * out_time + t) * dim + h;
                    let n = noise_amp * noise[idx];
                    noise_out[idx] = n;
                    sine_waves[idx] = phase[idx].sin() * self.sine_amp * uv_value + n;
                }
            }
        }

        let sine = Tensor::from_vec(sine_waves, (batch, out_time, dim), device)?;
        let uv = Tensor::from_vec(uv, (batch, time, 1), device)?;
        let noise = Tensor::from_vec(noise_out, (batch, out_time, dim), device)?;
        Ok((sine, uv, noise))
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

fn interpolate_linear_btd(
    values: &[f32],
    batch: usize,
    in_time: usize,
    dim: usize,
    scale_factor: f64,
) -> Vec<f32> {
    let out_time = ((in_time as f64) * scale_factor).floor() as usize;
    let mut out = vec![0.0f32; batch * out_time * dim];
    for b in 0..batch {
        for t_out in 0..out_time {
            let in_pos = ((t_out as f64 + 0.5) / scale_factor) - 0.5;
            let in_pos = in_pos.clamp(0.0, (in_time - 1) as f64);
            let t0 = in_pos.floor() as usize;
            let t1 = (t0 + 1).min(in_time - 1);
            let frac = (in_pos - t0 as f64) as f32;
            for h in 0..dim {
                let v0 = values[(b * in_time + t0) * dim + h];
                let v1 = values[(b * in_time + t1) * dim + h];
                out[(b * out_time + t_out) * dim + h] = v0 * (1.0 - frac) + v1 * frac;
            }
        }
    }
    out
}
