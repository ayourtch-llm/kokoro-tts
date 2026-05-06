#![allow(dead_code)]

use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

pub fn fold_weight_norm_conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: candle_nn::Conv1dConfig,
    bias: bool,
    vb: VarBuilder,
) -> Result<candle_nn::Conv1d> {
    if vb.contains_tensor("weight") {
        return if bias {
            candle_nn::conv1d(in_channels, out_channels, kernel_size, cfg, vb)
        } else {
            candle_nn::conv1d_no_bias(in_channels, out_channels, kernel_size, cfg, vb)
        };
    }

    let weight_g = if vb.contains_tensor("weight_g") {
        vb.get((out_channels, 1, 1), "weight_g")?
    } else {
        vb.get((out_channels, 1, 1), "parametrizations.weight.original0")?
    };
    let weight_v = if vb.contains_tensor("weight_v") {
        vb.get(
            (out_channels, in_channels / cfg.groups, kernel_size),
            "weight_v",
        )?
    } else {
        vb.get(
            (out_channels, in_channels / cfg.groups, kernel_size),
            "parametrizations.weight.original1",
        )?
    };
    let denom = weight_v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    let weight = weight_v.broadcast_div(&denom)?.broadcast_mul(&weight_g)?;
    let bias = if bias {
        Some(vb.get(out_channels, "bias")?)
    } else {
        None
    };
    Ok(candle_nn::Conv1d::new(weight, bias, cfg))
}

pub fn fold_weight_norm_conv_transpose1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: candle_nn::ConvTranspose1dConfig,
    vb: VarBuilder,
) -> Result<candle_nn::ConvTranspose1d> {
    if vb.contains_tensor("weight") {
        return candle_nn::conv_transpose1d(in_channels, out_channels, kernel_size, cfg, vb);
    }

    let weight_g = if vb.contains_tensor("weight_g") {
        vb.get((in_channels, 1, 1), "weight_g")?
    } else {
        vb.get((in_channels, 1, 1), "parametrizations.weight.original0")?
    };
    let weight_v = if vb.contains_tensor("weight_v") {
        vb.get(
            (in_channels, out_channels / cfg.groups, kernel_size),
            "weight_v",
        )?
    } else {
        vb.get(
            (in_channels, out_channels / cfg.groups, kernel_size),
            "parametrizations.weight.original1",
        )?
    };
    let denom = weight_v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    let weight = weight_v.broadcast_div(&denom)?.broadcast_mul(&weight_g)?;
    let bias = vb.get(out_channels, "bias")?;
    Ok(candle_nn::ConvTranspose1d::new(weight, Some(bias), cfg))
}

fn instance_norm1d(x: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(2)?;
    let var = x.broadcast_sub(&mean)?.sqr()?.mean_keepdim(2)?;
    x.broadcast_sub(&mean)?.broadcast_div(
        &var.broadcast_add(&Tensor::new(eps as f32, x.device())?)?
            .sqrt()?,
    )
}

/// AdaIN 1d for decoder
pub struct AdaIN1d {
    fc: candle_nn::Linear,
}

impl AdaIN1d {
    pub fn load(style_dim: usize, num_features: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc: candle_nn::linear(style_dim, num_features * 2, vb.pp("fc"))?,
        })
    }

    pub fn forward(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        let h = self.fc.forward(s)?;
        let h = h.unsqueeze(2)?;
        let chunks = h.chunk(2, 1)?;
        let gamma = &chunks[0];
        let beta = &chunks[1];
        let x = instance_norm1d(x, 1e-5)?;
        let one = gamma.ones_like()?;
        (gamma + one)?.broadcast_mul(&x)?.broadcast_add(beta)
    }
}

/// AdainResBlk1d for decoder
pub struct AdainResBlk1d {
    conv1: candle_nn::Conv1d,
    conv2: candle_nn::Conv1d,
    norm1: AdaIN1d,
    norm2: AdaIN1d,
    conv1x1: Option<candle_nn::Conv1d>,
    upsample: bool,
    pool: Option<candle_nn::ConvTranspose1d>,
}

impl AdainResBlk1d {
    pub fn load(
        dim_in: usize,
        dim_out: usize,
        style_dim: usize,
        upsample: bool,
        dropout_p: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let _ = dropout_p;
        let conv1 = fold_weight_norm_conv1d(
            dim_in,
            dim_out,
            3,
            candle_nn::Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            true,
            vb.pp("conv1"),
        )?;
        let conv2 = fold_weight_norm_conv1d(
            dim_out,
            dim_out,
            3,
            candle_nn::Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            true,
            vb.pp("conv2"),
        )?;
        let norm1 = AdaIN1d::load(style_dim, dim_in, vb.pp("norm1"))?;
        let norm2 = AdaIN1d::load(style_dim, dim_out, vb.pp("norm2"))?;
        let conv1x1 = if dim_in != dim_out {
            Some(fold_weight_norm_conv1d(
                dim_in,
                dim_out,
                1,
                Default::default(),
                false,
                vb.pp("conv1x1"),
            )?)
        } else {
            None
        };
        let pool = if upsample {
            Some(fold_weight_norm_conv_transpose1d(
                dim_in,
                dim_in,
                3,
                candle_nn::ConvTranspose1dConfig {
                    stride: 2,
                    groups: dim_in,
                    padding: 1,
                    output_padding: 1,
                    ..Default::default()
                },
                vb.pp("pool"),
            )?)
        } else {
            None
        };
        Ok(Self {
            conv1,
            conv2,
            norm1,
            norm2,
            conv1x1,
            upsample,
            pool,
        })
    }

    pub fn forward(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        let shortcut = self._shortcut(x)?;
        let residual = self._residual(x, s)?;
        let scale = (2.0_f64).sqrt().recip();
        (residual + shortcut)? * scale
    }

    fn _shortcut(&self, x: &Tensor) -> Result<Tensor> {
        let x = if self.upsample {
            x.upsample_nearest1d(x.dim(2)? * 2)?
        } else {
            x.clone()
        };
        if let Some(ref c) = self.conv1x1 {
            c.forward(&x)
        } else {
            Ok(x)
        }
    }

    fn _residual(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        let x = self.norm1.forward(x, s)?;
        let x = candle_nn::ops::leaky_relu(&x, 0.2)?;
        let x = if let Some(ref pool) = self.pool {
            pool.forward(&x)?
        } else {
            x.clone()
        };
        let x = self.conv1.forward(&x)?;
        let x = self.norm2.forward(&x, s)?;
        let x = candle_nn::ops::leaky_relu(&x, 0.2)?;
        self.conv2.forward(&x)
    }
}

/// iSTFTNet Decoder — front-end fusion + Generator vocoder.
pub struct Decoder {
    encode: AdainResBlk1d,
    decode: Vec<AdainResBlk1d>,
    f0_conv: candle_nn::Conv1d,
    n_conv: candle_nn::Conv1d,
    asr_res: candle_nn::Conv1d,
    generator: crate::model::generator::Generator,
    style_dim: usize,
}

impl Decoder {
    pub fn load(
        dim_in: usize,
        style_dim: usize,
        istftnet: &crate::model::config::IstftnetConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let encode = AdainResBlk1d::load(dim_in + 2, 1024, style_dim, false, 0.0, vb.pp("encode"))?;

        let mut decode = Vec::new();
        decode.push(AdainResBlk1d::load(
            1024 + 2 + 64,
            1024,
            style_dim,
            false,
            0.0,
            vb.pp("decode.0"),
        )?);
        decode.push(AdainResBlk1d::load(
            1024 + 2 + 64,
            1024,
            style_dim,
            false,
            0.0,
            vb.pp("decode.1"),
        )?);
        decode.push(AdainResBlk1d::load(
            1024 + 2 + 64,
            1024,
            style_dim,
            false,
            0.0,
            vb.pp("decode.2"),
        )?);
        decode.push(AdainResBlk1d::load(
            1024 + 2 + 64,
            512,
            style_dim,
            true,
            0.0,
            vb.pp("decode.3"),
        )?);

        let f0_conv = fold_weight_norm_conv1d(
            1,
            1,
            3,
            candle_nn::Conv1dConfig {
                stride: 2,
                padding: 1,
                groups: 1,
                ..Default::default()
            },
            true,
            vb.pp("F0_conv"),
        )?;
        let n_conv = fold_weight_norm_conv1d(
            1,
            1,
            3,
            candle_nn::Conv1dConfig {
                stride: 2,
                padding: 1,
                groups: 1,
                ..Default::default()
            },
            true,
            vb.pp("N_conv"),
        )?;
        let asr_res =
            fold_weight_norm_conv1d(512, 64, 1, Default::default(), true, vb.pp("asr_res.0"))?;

        let dilation_arrays: Vec<[usize; 3]> = istftnet
            .resblock_dilation_sizes
            .iter()
            .map(|d| {
                if d.len() != 3 {
                    return Err(candle_core::Error::Msg(format!(
                        "expected resblock dilations of length 3, got {}",
                        d.len()
                    )));
                }
                Ok([d[0], d[1], d[2]])
            })
            .collect::<Result<_>>()?;
        let generator = crate::model::generator::Generator::load(
            style_dim,
            istftnet.upsample_initial_channel,
            istftnet.upsample_rates.clone(),
            istftnet.upsample_kernel_sizes.clone(),
            istftnet.resblock_kernel_sizes.clone(),
            dilation_arrays,
            istftnet.gen_istft_n_fft,
            istftnet.gen_istft_hop_size,
            vb.pp("generator"),
        )?;

        Ok(Self {
            encode,
            decode,
            f0_conv,
            n_conv,
            asr_res,
            generator,
            style_dim,
        })
    }

    pub fn forward(
        &self,
        asr: &Tensor,
        f0_curve: &Tensor,
        n: &Tensor,
        s: &Tensor,
    ) -> Result<Tensor> {
        let x = self.forward_pre_generator(asr, f0_curve, n, s)?;
        // Pass the (un-strided) F0_curve to the generator — it does its own
        // F0 upsampling internally for the NSF source path.
        self.generator.forward(&x, s, f0_curve)
    }

    /// The pre-generator (decode-block-loop) output. Used by stage-9 validation
    /// which compares against the upstream Python reference taken at this point.
    pub fn forward_pre_generator(
        &self,
        asr: &Tensor,
        f0_curve: &Tensor,
        n: &Tensor,
        s: &Tensor,
    ) -> Result<Tensor> {
        let f0 = self.f0_conv.forward(&f0_curve.unsqueeze(1)?)?;
        let n = self.n_conv.forward(&n.unsqueeze(1)?)?;
        let x = Tensor::cat(&[asr, &f0, &n], 1)?;
        let x = self.encode.forward(&x, s)?;
        let asr_res = self.asr_res.forward(asr)?;

        let mut x = x.clone();
        let mut res = true;
        for block in &self.decode {
            if res {
                x = Tensor::cat(&[&x, &asr_res, &f0, &n], 1)?;
            }
            x = block.forward(&x, s)?;
            if block.upsample {
                res = false;
            }
        }
        Ok(x)
    }
}
