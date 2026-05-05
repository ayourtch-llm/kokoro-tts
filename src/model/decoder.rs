#![allow(dead_code)]

use candle_core::{DType, Result, Tensor};
use candle_nn::{Module, VarBuilder};

fn instance_norm1d(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(2)?;
    let var = x.broadcast_sub(&mean)?.sqr()?.mean_keepdim(2)?;
    let normalized = x.broadcast_sub(&mean)?.broadcast_div(
        &var.broadcast_add(&Tensor::new(eps as f32, x.device())?)?
            .sqrt()?,
    )?;
    normalized
        .broadcast_mul(&weight.reshape((1, weight.dim(0)?, 1))?)?
        .broadcast_add(&bias.reshape((1, bias.dim(0)?, 1))?)
}

/// AdaIN 1d for decoder
pub struct AdaIN1d {
    norm_weight: Tensor,
    norm_bias: Tensor,
    fc: candle_nn::Linear,
}

impl AdaIN1d {
    pub fn load(style_dim: usize, num_features: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm_weight: vb.get(num_features, "norm.weight")?,
            norm_bias: vb.get(num_features, "norm.bias")?,
            fc: candle_nn::linear(style_dim, num_features * 2, vb.pp("fc"))?,
        })
    }

    pub fn forward(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        let h = self.fc.forward(s)?;
        let h = h.unsqueeze(2)?;
        let chunks = h.chunk(2, 1)?;
        let gamma = &chunks[0];
        let beta = &chunks[1];
        let x = instance_norm1d(x, &self.norm_weight, &self.norm_bias, 1e-5)?;
        (((gamma + &Tensor::ones(gamma.shape(), DType::F32, x.device())?)?) * &x)? + beta
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
        let conv1 = candle_nn::conv1d(
            dim_in,
            dim_out,
            3,
            candle_nn::Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;
        let conv2 = candle_nn::conv1d(
            dim_out,
            dim_out,
            3,
            candle_nn::Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv2"),
        )?;
        let norm1 = AdaIN1d::load(style_dim, dim_in, vb.pp("norm1"))?;
        let norm2 = AdaIN1d::load(style_dim, dim_out, vb.pp("norm2"))?;
        let conv1x1 = if dim_in != dim_out {
            Some(candle_nn::conv1d_no_bias(
                dim_in,
                dim_out,
                1,
                Default::default(),
                vb.pp("conv1x1"),
            )?)
        } else {
            None
        };
        let pool = if upsample {
            Some(candle_nn::conv_transpose1d(
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
        let x = if let Some(ref pool) = self.pool {
            pool.forward(x)?
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

/// iSTFTNet Decoder (skeleton - full implementation needs Generator + STFT)
pub struct Decoder {
    encode: AdainResBlk1d,
    decode: Vec<AdainResBlk1d>,
    f0_conv: candle_nn::Conv1d,
    n_conv: candle_nn::Conv1d,
    asr_res: candle_nn::Conv1d,
    style_dim: usize,
}

impl Decoder {
    pub fn load(dim_in: usize, style_dim: usize, vb: VarBuilder) -> Result<Self> {
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

        let f0_conv = candle_nn::conv1d(
            1,
            1,
            3,
            candle_nn::Conv1dConfig {
                stride: 2,
                padding: 1,
                groups: 1,
                ..Default::default()
            },
            vb.pp("f0_conv"),
        )?;
        let n_conv = candle_nn::conv1d(
            1,
            1,
            3,
            candle_nn::Conv1dConfig {
                stride: 2,
                padding: 1,
                groups: 1,
                ..Default::default()
            },
            vb.pp("n_conv"),
        )?;
        let asr_res = candle_nn::conv1d(512, 64, 1, Default::default(), vb.pp("asr_res"))?;

        Ok(Self {
            encode,
            decode,
            f0_conv,
            n_conv,
            asr_res,
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

        // TODO: Generator (multi-scale upsampling + iSTFT) needed here
        // This is the most complex part - requires NSF source, multi-band STFT
        Ok(x)
    }
}
