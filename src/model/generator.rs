#![allow(dead_code)]

use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

use crate::model::decoder::{fold_weight_norm_conv1d, AdaIN1d};

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
