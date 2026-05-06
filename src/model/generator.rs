#![allow(dead_code)]

use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

use crate::model::decoder::{fold_weight_norm_conv1d, fold_weight_norm_conv_transpose1d, AdaIN1d};
use crate::model::source::{SourceModuleHnNsf, KOKORO_UPSAMPLE_SCALE};
use crate::model::stft::CustomStft;

pub struct Snake1d {
    alpha: Tensor,
}

impl Snake1d {
    pub fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            alpha: vb.get((1, channels, 1), "")?,
        })
    }

    pub fn load_named(channels: usize, vb: VarBuilder, name: &str) -> Result<Self> {
        Ok(Self {
            alpha: vb.get((1, channels, 1), name)?,
        })
    }

    pub fn new(alpha: Tensor) -> Self {
        Self { alpha }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        snake1d(x, &self.alpha)
    }
}

pub fn snake1d(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    let scaled = alpha.broadcast_mul(x)?;
    let sine_sq = scaled.sin()?.sqr()?;
    x + alpha.recip()?.broadcast_mul(&sine_sq)?
}

/// AdaINResBlock1 — the 3-sub-block residual block with Snake1D activation
/// used inside `Generator.resblocks` and `Generator.noise_res`. Distinct from
/// `decoder::AdainResBlk1d` (which is the StyleTTS2-style 2-conv block used by
/// the predictor and the decoder front-end).
///
/// Layout per upstream `kokoro/istftnet.py:34-77`:
/// - convs1.{0,1,2}: dilated Conv1d (weight-normed)
/// - convs2.{0,1,2}: plain Conv1d (weight-normed)
/// - adain1.{0,1,2}, adain2.{0,1,2}: AdaIN1d
/// - alpha1.{0,1,2}, alpha2.{0,1,2}: bare nn.Parameter [1, channels, 1]
///
/// Forward: for each j in 0..3:
///   xt = adain1[j](x, s) → snake1d(xt, alpha1[j]) → convs1[j](xt)
///        → adain2[j](xt, s) → snake1d(xt, alpha2[j]) → convs2[j](xt)
///   x = xt + x   (residual at every sub-block, not at the end)
pub struct AdaINResBlock1 {
    convs1: Vec<candle_nn::Conv1d>,
    convs2: Vec<candle_nn::Conv1d>,
    adain1: Vec<AdaIN1d>,
    adain2: Vec<AdaIN1d>,
    alpha1: Vec<Tensor>,
    alpha2: Vec<Tensor>,
}

impl AdaINResBlock1 {
    pub fn load(
        channels: usize,
        kernel_size: usize,
        dilations: [usize; 3],
        style_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut convs1 = Vec::with_capacity(3);
        let mut convs2 = Vec::with_capacity(3);
        let mut adain1 = Vec::with_capacity(3);
        let mut adain2 = Vec::with_capacity(3);
        let mut alpha1 = Vec::with_capacity(3);
        let mut alpha2 = Vec::with_capacity(3);

        for j in 0..3 {
            let dil = dilations[j];
            // upstream get_padding(k, d) = int((k*d - d) / 2) = (k-1)*d/2 for k odd
            let pad1 = (kernel_size * dil - dil) / 2;
            let pad2 = (kernel_size - 1) / 2;

            convs1.push(fold_weight_norm_conv1d(
                channels,
                channels,
                kernel_size,
                candle_nn::Conv1dConfig {
                    padding: pad1,
                    dilation: dil,
                    ..Default::default()
                },
                true,
                vb.pp(format!("convs1.{j}")),
            )?);
            convs2.push(fold_weight_norm_conv1d(
                channels,
                channels,
                kernel_size,
                candle_nn::Conv1dConfig {
                    padding: pad2,
                    dilation: 1,
                    ..Default::default()
                },
                true,
                vb.pp(format!("convs2.{j}")),
            )?);
            adain1.push(AdaIN1d::load(
                style_dim,
                channels,
                vb.pp(format!("adain1.{j}")),
            )?);
            adain2.push(AdaIN1d::load(
                style_dim,
                channels,
                vb.pp(format!("adain2.{j}")),
            )?);
            alpha1.push(vb.get((1, channels, 1), &format!("alpha1.{j}"))?);
            alpha2.push(vb.get((1, channels, 1), &format!("alpha2.{j}"))?);
        }

        Ok(Self {
            convs1,
            convs2,
            adain1,
            adain2,
            alpha1,
            alpha2,
        })
    }

    pub fn forward(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for j in 0..3 {
            let xt = self.adain1[j].forward(&x, s)?;
            let xt = snake1d(&xt, &self.alpha1[j])?;
            let xt = self.convs1[j].forward(&xt)?;
            let xt = self.adain2[j].forward(&xt, s)?;
            let xt = snake1d(&xt, &self.alpha2[j])?;
            let xt = self.convs2[j].forward(&xt)?;
            x = (xt + x)?;
        }
        Ok(x)
    }
}

/// Reflection padding on the left of the last (time) dimension.
/// `nn.ReflectionPad1d((pad, 0))` equivalent — pads `pad` frames on left,
/// 0 on right, mirroring across index 0 (so for [a, b, c] with pad=1
/// you get [b, a, b, c]; reflection skips the boundary itself).
fn reflection_pad_1d_left(x: &Tensor, pad: usize) -> Result<Tensor> {
    if pad == 0 {
        return Ok(x.clone());
    }
    let last_dim = x.rank() - 1;
    let len = x.dim(last_dim)?;
    if pad >= len {
        return Err(candle_core::Error::Msg(format!(
            "reflection_pad_1d_left: pad {pad} >= length {len}"
        )));
    }
    // For pad=1: reflect index 1. For pad=k: reflect indices [1..=k] in reverse.
    let mut parts = Vec::with_capacity(pad + 1);
    for i in (1..=pad).rev() {
        parts.push(x.narrow(last_dim, i, 1)?);
    }
    parts.push(x.clone());
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, last_dim)
}

/// iSTFTNet Generator — vocoder that turns post-decode features + style + F0
/// into a waveform via NSF source + multi-scale upsampling + iSTFT.
///
/// Per upstream `kokoro/istftnet.py:257-326` and Kokoro-82M config.
/// State-dict prefix: `decoder.generator.<...>` (303 keys total).
pub struct Generator {
    m_source: SourceModuleHnNsf,
    stft: CustomStft,
    noise_convs: Vec<candle_nn::Conv1d>,
    noise_res: Vec<AdaINResBlock1>,
    ups: Vec<candle_nn::ConvTranspose1d>,
    resblocks: Vec<AdaINResBlock1>,
    conv_post: candle_nn::Conv1d,
    upsample_rates: Vec<usize>,
    num_kernels: usize,
    n_fft: usize,
    f0_upsample_factor: usize,
}

impl Generator {
    pub fn load(
        style_dim: usize,
        upsample_initial_channel: usize,
        upsample_rates: Vec<usize>,
        upsample_kernel_sizes: Vec<usize>,
        resblock_kernel_sizes: Vec<usize>,
        resblock_dilation_sizes: Vec<[usize; 3]>,
        n_fft: usize,
        hop_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let device = vb.device().clone();
        let num_upsamples = upsample_rates.len();
        let num_kernels = resblock_kernel_sizes.len();

        let m_source = SourceModuleHnNsf::load(vb.pp("m_source"))?;
        let stft = CustomStft::new(n_fft, hop_size, &device)?;

        // ups: ConvTranspose1d, weight-normed
        let mut ups = Vec::with_capacity(num_upsamples);
        for i in 0..num_upsamples {
            let in_ch = upsample_initial_channel / (1 << i);
            let out_ch = upsample_initial_channel / (1 << (i + 1));
            let k = upsample_kernel_sizes[i];
            let stride = upsample_rates[i];
            let padding = (k - stride) / 2;
            ups.push(fold_weight_norm_conv_transpose1d(
                in_ch,
                out_ch,
                k,
                candle_nn::ConvTranspose1dConfig {
                    stride,
                    padding,
                    output_padding: 0,
                    dilation: 1,
                    groups: 1,
                },
                vb.pp(format!("ups.{i}")),
            )?);
        }

        // noise_convs (plain Conv1d, NOT weight-normed) and noise_res (AdaINResBlock1)
        let mut noise_convs = Vec::with_capacity(num_upsamples);
        let mut noise_res = Vec::with_capacity(num_upsamples);
        let mut resblocks = Vec::with_capacity(num_upsamples * num_kernels);
        for i in 0..num_upsamples {
            let ch = upsample_initial_channel / (1 << (i + 1));
            // resblocks for this stage
            for j in 0..num_kernels {
                resblocks.push(AdaINResBlock1::load(
                    ch,
                    resblock_kernel_sizes[j],
                    resblock_dilation_sizes[j],
                    style_dim,
                    vb.pp(format!("resblocks.{}", i * num_kernels + j)),
                )?);
            }
            if i + 1 < num_upsamples {
                let stride_f0: usize = upsample_rates[i + 1..].iter().product();
                let nk = stride_f0 * 2;
                let np = (stride_f0 + 1) / 2;
                noise_convs.push(candle_nn::conv1d(
                    n_fft + 2,
                    ch,
                    nk,
                    candle_nn::Conv1dConfig {
                        stride: stride_f0,
                        padding: np,
                        ..Default::default()
                    },
                    vb.pp(format!("noise_convs.{i}")),
                )?);
                noise_res.push(AdaINResBlock1::load(
                    ch,
                    7,
                    [1, 3, 5],
                    style_dim,
                    vb.pp(format!("noise_res.{i}")),
                )?);
            } else {
                // last upsample stage: kernel=1, no downsample
                noise_convs.push(candle_nn::conv1d(
                    n_fft + 2,
                    ch,
                    1,
                    candle_nn::Conv1dConfig::default(),
                    vb.pp(format!("noise_convs.{i}")),
                )?);
                noise_res.push(AdaINResBlock1::load(
                    ch,
                    11,
                    [1, 3, 5],
                    style_dim,
                    vb.pp(format!("noise_res.{i}")),
                )?);
            }
        }

        // conv_post: weight-normed Conv1d(final_ch, n_fft + 2, kernel=7, padding=3)
        let final_ch = upsample_initial_channel / (1 << num_upsamples);
        let conv_post = fold_weight_norm_conv1d(
            final_ch,
            n_fft + 2,
            7,
            candle_nn::Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            true,
            vb.pp("conv_post"),
        )?;

        let f0_upsample_factor: usize = upsample_rates.iter().product::<usize>() * hop_size;

        Ok(Self {
            m_source,
            stft,
            noise_convs,
            noise_res,
            ups,
            resblocks,
            conv_post,
            upsample_rates,
            num_kernels,
            n_fft,
            f0_upsample_factor,
        })
    }

    /// Stochastic-control variant for validation against a Python reference.
    /// `rand_ini` and `noise` correspond to the SineGen rand_ini / standard_noise
    /// arguments — pass deterministic tensors to make the output reproducible.
    pub fn forward_with_controls(
        &self,
        x: &Tensor,
        s: &Tensor,
        f0: &Tensor,
        rand_ini: Option<&Tensor>,
        noise: Option<&Tensor>,
    ) -> Result<Tensor> {
        // f0: [B, T_f0]   (note: F0_curve from predictor is [B, 2*T_dec])
        // f0_upsample_factor = prod(upsample_rates) * hop_size  (= 300 for Kokoro)
        let f0_audio_len = f0.dim(1)? * self.f0_upsample_factor;
        let f0_up = f0
            .unsqueeze(1)? // [B, 1, T_f0]
            .upsample_nearest1d(f0_audio_len)? // [B, 1, T_audio]
            .transpose(1, 2)?; // [B, T_audio, 1]

        // NSF source: [B, T_audio, 1] sine_merge → squeeze → [B, T_audio]
        let (har_source_3d, _, _) = self
            .m_source
            .forward_with_controls(&f0_up, rand_ini, noise)?;
        let har_source = har_source_3d.transpose(1, 2)?.squeeze(1)?;
        let (har_spec, har_phase) = self.stft.transform(&har_source)?;
        let har = Tensor::cat(&[&har_spec, &har_phase], 1)?;

        let num_upsamples = self.ups.len();
        let mut x = x.clone();
        for i in 0..num_upsamples {
            let activated = candle_nn::ops::leaky_relu(&x, 0.1)?;
            let x_source = self.noise_convs[i].forward(&har)?;
            let x_source = self.noise_res[i].forward(&x_source, s)?;
            let mut up = self.ups[i].forward(&activated)?;
            if i == num_upsamples - 1 {
                up = reflection_pad_1d_left(&up, 1)?;
            }
            let combined = up.broadcast_add(&x_source)?;
            // MRF: average resblock outputs across kernel sizes
            let mut accum: Option<Tensor> = None;
            for j in 0..self.num_kernels {
                let xs_j = self.resblocks[i * self.num_kernels + j].forward(&combined, s)?;
                accum = Some(match accum {
                    Some(a) => (a + xs_j)?,
                    None => xs_j,
                });
            }
            x = (accum.unwrap() / (self.num_kernels as f64))?;
        }

        // Post: F.leaky_relu (default slope = 0.01) → conv_post → split → exp/sin → iSTFT
        let x = candle_nn::ops::leaky_relu(&x, 0.01)?;
        let x = self.conv_post.forward(&x)?;
        let n_freq = self.n_fft / 2 + 1;
        let spec = x.narrow(1, 0, n_freq)?.exp()?;
        let phase = x.narrow(1, n_freq, n_freq)?.sin()?;
        self.stft.inverse(&spec, &phase, None)
    }

    /// Default forward: zero rand_ini, zero noise. Useful for sanity checks
    /// but not for matching a stochastic Python reference.
    pub fn forward(&self, x: &Tensor, s: &Tensor, f0: &Tensor) -> Result<Tensor> {
        self.forward_with_controls(x, s, f0, None, None)
    }

    /// Debug variant that writes intermediate tensors to `dump_dir` for diff vs Python reference.
    pub fn debug_forward(
        &self,
        x: &Tensor,
        s: &Tensor,
        f0: &Tensor,
        rand_ini: Option<&Tensor>,
        noise: Option<&Tensor>,
        dump_dir: &std::path::Path,
    ) -> Result<Tensor> {
        let f0_audio_len = f0.dim(1)? * self.f0_upsample_factor;
        let f0_up = f0
            .unsqueeze(1)?
            .upsample_nearest1d(f0_audio_len)?
            .transpose(1, 2)?;
        let (har_source_3d, _, _) = self
            .m_source
            .forward_with_controls(&f0_up, rand_ini, noise)?;
        let har_source = har_source_3d.transpose(1, 2)?.squeeze(1)?;
        let (har_spec, har_phase) = self.stft.transform(&har_source)?;
        let har = Tensor::cat(&[&har_spec, &har_phase], 1)?;
        dump_tensor(dump_dir, "_har.bin", &har)?;

        let num_upsamples = self.ups.len();
        let mut x = x.clone();
        for i in 0..num_upsamples {
            let activated = candle_nn::ops::leaky_relu(&x, 0.1)?;
            let x_source = self.noise_convs[i].forward(&har)?;
            let x_source = self.noise_res[i].forward(&x_source, s)?;
            let mut up = self.ups[i].forward(&activated)?;
            if i == num_upsamples - 1 {
                up = reflection_pad_1d_left(&up, 1)?;
            }
            let combined = up.broadcast_add(&x_source)?;
            let mut accum: Option<Tensor> = None;
            for j in 0..self.num_kernels {
                let xs_j = self.resblocks[i * self.num_kernels + j].forward(&combined, s)?;
                accum = Some(match accum {
                    Some(a) => (a + xs_j)?,
                    None => xs_j,
                });
            }
            x = (accum.unwrap() / (self.num_kernels as f64))?;
            dump_tensor(dump_dir, &format!("_stage_{i}.bin"), &x)?;
        }

        let x = candle_nn::ops::leaky_relu(&x, 0.01)?;
        let x = self.conv_post.forward(&x)?;
        dump_tensor(dump_dir, "_conv_post.bin", &x)?;
        let n_freq = self.n_fft / 2 + 1;
        let spec = x.narrow(1, 0, n_freq)?.exp()?;
        let phase = x.narrow(1, n_freq, n_freq)?.sin()?;
        self.stft.inverse(&spec, &phase, None)
    }
}

fn dump_tensor(dir: &std::path::Path, name: &str, t: &Tensor) -> Result<()> {
    use std::io::Write;
    std::fs::create_dir_all(dir).ok();
    let path = dir.join(name);
    let dims = t.dims().to_vec();
    let data: Vec<f32> = t
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1()?;
    let mut file = std::fs::File::create(&path)
        .map_err(|e| candle_core::Error::Msg(format!("create {}: {e}", path.display())))?;
    let ndim = dims.len() as u32;
    file.write_all(&ndim.to_le_bytes()).ok();
    for d in &dims {
        file.write_all(&(*d as u32).to_le_bytes()).ok();
    }
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    file.write_all(&bytes).ok();
    Ok(())
}

// Re-export the Kokoro upsample scale for callers that need to size F0
// upsampling without recomputing prod(upsample_rates) * hop_size.
pub const KOKORO_GENERATOR_F0_UPSAMPLE: usize = KOKORO_UPSAMPLE_SCALE;
