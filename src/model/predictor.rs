#![allow(dead_code)]

use candle_core::{DType, IndexOp, Result, Tensor};
use candle_nn::rnn::Direction;
use candle_nn::{LSTMConfig, Module, VarBuilder, LSTM, RNN};

fn fold_weight_norm_conv1d(
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

fn fold_weight_norm_conv_transpose1d(
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

fn leaky_relu(x: &Tensor) -> Result<Tensor> {
    candle_nn::ops::leaky_relu(x, 0.2)
}

fn instance_norm1d(x: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(2)?;
    let var = x.broadcast_sub(&mean)?.sqr()?.mean_keepdim(2)?;
    x.broadcast_sub(&mean)?.broadcast_div(
        &var.broadcast_add(&Tensor::new(eps as f32, x.device())?)?
            .sqrt()?,
    )
}

fn layer_norm_last_dim(x: &Tensor, eps: f64) -> Result<Tensor> {
    let last_dim = x.rank() - 1;
    let mean = x.mean_keepdim(last_dim)?;
    let var = x.broadcast_sub(&mean)?.sqr()?.mean_keepdim(last_dim)?;
    x.broadcast_sub(&mean)?.broadcast_div(
        &var.broadcast_add(&Tensor::new(eps as f32, x.device())?)?
            .sqrt()?,
    )
}

struct BiLstm {
    forward: LSTM,
    backward: LSTM,
}

impl BiLstm {
    fn load(in_dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        let forward = candle_nn::lstm(in_dim, hidden_dim, LSTMConfig::default(), vb.clone())?;
        let backward = candle_nn::lstm(
            in_dim,
            hidden_dim,
            LSTMConfig {
                direction: Direction::Backward,
                ..Default::default()
            },
            vb,
        )?;
        Ok(Self { forward, backward })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let fw = self.forward.seq(x)?;
        let fw = self.forward.states_to_tensor(&fw)?;

        let (_, seq_len, _) = x.dims3()?;
        let mut reversed = Vec::with_capacity(seq_len);
        for i in (0..seq_len).rev() {
            reversed.push(x.i((.., i, ..))?);
        }
        let rev = Tensor::stack(&reversed, 1)?;
        let bw = self.backward.seq(&rev)?;
        let bw = self.backward.states_to_tensor(&bw)?;
        let mut restored = Vec::with_capacity(seq_len);
        for i in (0..seq_len).rev() {
            restored.push(bw.i((.., i, ..))?);
        }
        let bw = Tensor::stack(&restored, 1)?;
        Tensor::cat(&[&fw, &bw], 2)
    }
}

/// AdaLayerNorm: style-adaptive layer normalization
pub struct AdaLayerNorm {
    fc: candle_nn::Linear,
    eps: f64,
}

impl AdaLayerNorm {
    pub fn load(style_dim: usize, channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc: candle_nn::linear(style_dim, channels * 2, vb.pp("fc"))?,
            eps: 1e-5,
        })
    }

    pub fn forward(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        let h = self.fc.forward(s)?.unsqueeze(1)?;
        let chunks = h.chunk(2, 2)?;
        let gamma = &chunks[0];
        let beta = &chunks[1];
        let x = layer_norm_last_dim(x, self.eps)?;
        let one = Tensor::ones(gamma.shape(), DType::F32, x.device())?;
        x.broadcast_mul(&(gamma + &one)?)?.broadcast_add(beta)
    }
}

/// LinearNorm: Linear with Xavier init
pub struct LinearNorm {
    linear: candle_nn::Linear,
}

impl LinearNorm {
    pub fn load(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear: candle_nn::linear(in_dim, out_dim, vb.pp("linear_layer"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x)
    }
}

/// DurationEncoder: LSTM stack with AdaLayerNorm
pub struct DurationEncoder {
    lstms: Vec<(BiLstm, AdaLayerNorm)>,
    dropout: f64,
}

impl DurationEncoder {
    pub fn load(
        style_dim: usize,
        d_model: usize,
        nlayers: usize,
        dropout: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut lstms = Vec::new();
        for i in 0..nlayers {
            let lstm = BiLstm::load(
                d_model + style_dim,
                d_model / 2,
                vb.pp(format!("lstms.{}", i * 2)),
            )?;
            let norm =
                AdaLayerNorm::load(style_dim, d_model, vb.pp(format!("lstms.{}", i * 2 + 1)))?;
            lstms.push((lstm, norm));
        }
        Ok(Self { lstms, dropout })
    }

    pub fn forward(&self, x: &Tensor, style: &Tensor, text_mask: &Tensor) -> Result<Tensor> {
        let mut x = x.transpose(1, 2)?; // [B, T, C]
        let s = style
            .unsqueeze(1)?
            .broadcast_as((x.dim(0)?, x.dim(1)?, style.dim(1)?))?;

        for (lstm, norm) in &self.lstms {
            x = Tensor::cat(&[&x, &s], 2)?;
            let zeros = x.zeros_like()?;
            let mask = text_mask.unsqueeze(2)?.broadcast_as(x.shape())?;
            x = mask.where_cond(&zeros, &x)?;
            x = lstm.forward(&x)?;
            let _ = self.dropout;
            x = norm.forward(&x, style)?;
        }

        x.transpose(1, 2) // [B, C, T]
    }
}

/// Duration-only subset of ProsodyPredictor used by stage validation.
pub struct DurationPredictor {
    text_encoder: DurationEncoder,
    lstm: BiLstm,
    duration_proj: LinearNorm,
}

impl DurationPredictor {
    pub fn load(
        style_dim: usize,
        d_hid: usize,
        nlayers: usize,
        max_dur: usize,
        dropout: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            text_encoder: DurationEncoder::load(
                style_dim,
                d_hid,
                nlayers,
                dropout,
                vb.pp("text_encoder"),
            )?,
            lstm: BiLstm::load(d_hid + style_dim, d_hid / 2, vb.pp("lstm"))?,
            duration_proj: LinearNorm::load(d_hid, max_dur, vb.pp("duration_proj"))?,
        })
    }

    pub fn predict_duration(
        &self,
        d_en: &Tensor,
        style: &Tensor,
        text_mask: &Tensor,
    ) -> Result<Tensor> {
        let d = self.text_encoder.forward(d_en, style, text_mask)?;
        let d = d.transpose(1, 2)?;
        let s = style
            .unsqueeze(1)?
            .broadcast_as((d.dim(0)?, d.dim(1)?, style.dim(1)?))?;
        let d = Tensor::cat(&[&d, &s], 2)?;
        let x = self.lstm.forward(&d)?;
        self.duration_proj.forward(&x)
    }
}

/// AdaIN 1d
pub struct Adain1d {
    fc: candle_nn::Linear,
}

impl Adain1d {
    fn load(style_dim: usize, num_features: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc: candle_nn::linear(style_dim, num_features * 2, vb.pp("fc"))?,
        })
    }

    fn forward(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        let h = self.fc.forward(s)?.unsqueeze(2)?;
        let chunks = h.chunk(2, 1)?;
        let gamma = &chunks[0];
        let beta = &chunks[1];
        let x = instance_norm1d(x, 1e-5)?;
        let one = gamma.ones_like()?;
        (gamma + one)?.broadcast_mul(&x)?.broadcast_add(beta)
    }
}

/// AdaIN ResBlock 1d
pub struct AdainResBlk1d {
    conv1: candle_nn::Conv1d,
    conv2: candle_nn::Conv1d,
    norm1: Adain1d,
    norm2: Adain1d,
    conv1x1: Option<candle_nn::Conv1d>,
    upsample: bool,
    pool: Option<candle_nn::ConvTranspose1d>,
}

impl AdainResBlk1d {
    fn load(
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
        let norm1 = Adain1d::load(style_dim, dim_in, vb.pp("norm1"))?;
        let norm2 = Adain1d::load(style_dim, dim_out, vb.pp("norm2"))?;
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

    fn forward(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        let residual = self._residual(x, s)?;
        let shortcut = self._shortcut(x)?;
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
        let residual = self.norm1.forward(x, s)?;
        let residual = leaky_relu(&residual)?;
        let residual = if let Some(ref pool) = self.pool {
            pool.forward(&residual)?
        } else {
            residual
        };
        let residual = self.conv1.forward(&residual)?;
        let residual = self.norm2.forward(&residual, s)?;
        let residual = leaky_relu(&residual)?;
        self.conv2.forward(&residual)
    }
}

/// ProsodyPredictor: predicts duration, F0, and noise
pub struct ProsodyPredictor {
    text_encoder: DurationEncoder,
    lstm: BiLstm,
    duration_proj: LinearNorm,
    shared: BiLstm,
    f0_blocks: Vec<AdainResBlk1d>,
    n_blocks: Vec<AdainResBlk1d>,
    f0_proj: candle_nn::Conv1d,
    n_proj: candle_nn::Conv1d,
}

impl ProsodyPredictor {
    pub fn load(
        style_dim: usize,
        d_hid: usize,
        nlayers: usize,
        max_dur: usize,
        dropout: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let text_encoder =
            DurationEncoder::load(style_dim, d_hid, nlayers, dropout, vb.pp("text_encoder"))?;
        let lstm = BiLstm::load(d_hid + style_dim, d_hid / 2, vb.pp("lstm"))?;
        let duration_proj = LinearNorm::load(d_hid, max_dur, vb.pp("duration_proj"))?;
        let shared = BiLstm::load(d_hid + style_dim, d_hid / 2, vb.pp("shared"))?;

        let f0_blocks = vec![
            AdainResBlk1d::load(d_hid, d_hid, style_dim, false, dropout, vb.pp("F0.0"))?,
            AdainResBlk1d::load(d_hid, d_hid / 2, style_dim, true, dropout, vb.pp("F0.1"))?,
            AdainResBlk1d::load(
                d_hid / 2,
                d_hid / 2,
                style_dim,
                false,
                dropout,
                vb.pp("F0.2"),
            )?,
        ];

        let n_blocks = vec![
            AdainResBlk1d::load(d_hid, d_hid, style_dim, false, dropout, vb.pp("N.0"))?,
            AdainResBlk1d::load(d_hid, d_hid / 2, style_dim, true, dropout, vb.pp("N.1"))?,
            AdainResBlk1d::load(
                d_hid / 2,
                d_hid / 2,
                style_dim,
                false,
                dropout,
                vb.pp("N.2"),
            )?,
        ];

        let f0_proj = candle_nn::conv1d(d_hid / 2, 1, 1, Default::default(), vb.pp("F0_proj"))?;
        let n_proj = candle_nn::conv1d(d_hid / 2, 1, 1, Default::default(), vb.pp("N_proj"))?;

        Ok(Self {
            text_encoder,
            lstm,
            duration_proj,
            shared,
            f0_blocks,
            n_blocks,
            f0_proj,
            n_proj,
        })
    }

    /// Forward for duration prediction
    pub fn predict_duration(
        &self,
        d_en: &Tensor,
        style: &Tensor,
        text_mask: &Tensor,
    ) -> Result<Tensor> {
        let d = self.text_encode(d_en, style, text_mask)?;
        self.duration_from_features(&d, style)
    }

    /// Run the predictor's text encoder. Returns the intermediate features `d`
    /// of shape `[B, hidden_dim, T]` — needed for both duration and F0/N paths.
    pub fn text_encode(&self, d_en: &Tensor, style: &Tensor, text_mask: &Tensor) -> Result<Tensor> {
        self.text_encoder.forward(d_en, style, text_mask)
    }

    /// Project text-encoded features → max_dur logits per token.
    /// Input `d` has shape `[B, hidden_dim, T]` (output of `text_encode`).
    pub fn duration_from_features(&self, d: &Tensor, style: &Tensor) -> Result<Tensor> {
        let d = d.transpose(1, 2)?; // [B, T, C]
        let s = style
            .unsqueeze(1)?
            .broadcast_as((d.dim(0)?, d.dim(1)?, style.dim(1)?))?;
        let d = Tensor::cat(&[&d, &s], 2)?;
        let x = self.lstm.forward(&d)?;
        self.duration_proj.forward(&x)
    }

    /// Forward for F0 and N prediction
    pub fn fn_train(&self, en: &Tensor, s: &Tensor) -> Result<(Tensor, Tensor)> {
        let en = en.transpose(1, 2)?;
        let style = s
            .unsqueeze(1)?
            .broadcast_as((en.dim(0)?, en.dim(1)?, s.dim(1)?))?;
        let en = Tensor::cat(&[&en, &style], 2)?;
        let x = self.shared.forward(&en)?;

        let mut f0 = x.transpose(1, 2)?;
        for block in &self.f0_blocks {
            f0 = block.forward(&f0, s)?;
        }
        let f0 = self.f0_proj.forward(&f0)?.squeeze(1)?;

        let mut n = x.transpose(1, 2)?;
        for block in &self.n_blocks {
            n = block.forward(&n, s)?;
        }
        let n = self.n_proj.forward(&n)?.squeeze(1)?;

        Ok((f0, n))
    }
}
