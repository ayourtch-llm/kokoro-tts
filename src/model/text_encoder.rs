#![allow(dead_code)]

use candle_core::{IndexOp, Module, Result, Tensor};
use candle_nn::rnn::Direction;
use candle_nn::{
    conv1d, embedding, lstm, Conv1d, Conv1dConfig, Embedding, LSTMConfig, VarBuilder, LSTM, RNN,
};

fn fold_weight_norm_conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    if vb.contains_tensor("weight") {
        return conv1d(in_channels, out_channels, kernel_size, cfg, vb);
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
    let bias = vb.get(out_channels, "bias")?;
    Ok(Conv1d::new(weight, Some(bias), cfg))
}

fn channel_layer_norm(x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f64) -> Result<Tensor> {
    let channel_last = x.transpose(1, x.rank() - 1)?;
    let mean = channel_last.mean_keepdim(channel_last.rank() - 1)?;
    let var = channel_last
        .broadcast_sub(&mean)?
        .sqr()?
        .mean_keepdim(channel_last.rank() - 1)?;
    let normalized = channel_last
        .broadcast_sub(&mean)?
        .broadcast_div(&var.affine(1.0, eps)?.sqrt()?)?;
    let y = normalized.broadcast_mul(gamma)?.broadcast_add(beta)?;
    y.transpose(1, y.rank() - 1)
}

struct ChannelLayerNorm {
    gamma: Tensor,
    beta: Tensor,
    eps: f64,
}

impl ChannelLayerNorm {
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gamma: vb.get(channels, "gamma")?,
            beta: vb.get(channels, "beta")?,
            eps: 1e-5,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        channel_layer_norm(x, &self.gamma, &self.beta, self.eps)
    }
}

/// CNN block: weight-normalized Conv1d -> channel LayerNorm -> LeakyReLU -> Dropout
struct CnnBlock {
    conv: Conv1d,
    norm: ChannelLayerNorm,
    dropout_p: f64,
}

impl CnnBlock {
    fn load(channels: usize, kernel_size: usize, dropout_p: f64, vb: VarBuilder) -> Result<Self> {
        let conv = fold_weight_norm_conv1d(
            channels,
            channels,
            kernel_size,
            Conv1dConfig {
                padding: (kernel_size - 1) / 2,
                ..Default::default()
            },
            vb.pp("0"),
        )?;
        let norm = ChannelLayerNorm::load(channels, vb.pp("1"))?;
        Ok(Self {
            conv,
            norm,
            dropout_p,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        let x = self.norm.forward(&x)?;
        let x = candle_nn::ops::leaky_relu(&x, 0.2)?;
        let _ = self.dropout_p;
        Ok(x)
    }
}

struct BiLstm {
    forward: LSTM,
    backward: LSTM,
}

impl BiLstm {
    fn load(in_dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        let forward = lstm(in_dim, hidden_dim, LSTMConfig::default(), vb.clone())?;
        let backward = lstm(
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

/// TextEncoder: Embedding -> CNNxN -> LSTM
pub struct TextEncoder {
    embedding: Embedding,
    cnns: Vec<CnnBlock>,
    lstm: BiLstm,
}

impl TextEncoder {
    pub fn load(
        channels: usize,
        kernel_size: usize,
        depth: usize,
        n_symbols: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let embedding = embedding(n_symbols, channels, vb.pp("embedding"))?;
        let mut cnns = Vec::new();
        for i in 0..depth {
            cnns.push(CnnBlock::load(
                channels,
                kernel_size,
                0.2,
                vb.pp(format!("cnn.{i}")),
            )?)
        }
        let lstm = BiLstm::load(channels, channels / 2, vb.pp("lstm"))?;
        Ok(Self {
            embedding,
            cnns,
            lstm,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, T] -> embedding -> [B, T, C] -> transpose -> [B, C, T]
        let x = self.embedding.forward(x)?;
        let x = x.transpose(1, 2)?;
        let mut x = x.clone();
        for cnn in &self.cnns {
            x = cnn.forward(&x)?;
        }
        // x: [B, C, T] -> transpose -> [B, T, C] for LSTM
        let x = x.transpose(1, 2)?;
        let x = self.lstm.forward(&x)?;
        // x: [B, T, C] -> transpose -> [B, C, T]
        x.transpose(1, 2)
    }
}
