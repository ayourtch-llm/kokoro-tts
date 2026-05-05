#![allow(dead_code)]

use candle_core::{IndexOp, Module, Result, Tensor};
use candle_nn::rnn::Direction;
use candle_nn::{
    conv1d, embedding, layer_norm, lstm, Conv1d, Conv1dConfig, Embedding, LSTMConfig, LayerNorm,
    VarBuilder, LSTM, RNN,
};

/// CNN block: Conv1d -> LayerNorm -> LeakyReLU -> Dropout
struct CnnBlock {
    conv: Conv1d,
    norm: LayerNorm,
    dropout_p: f64,
}

impl CnnBlock {
    fn load(channels: usize, kernel_size: usize, dropout_p: f64, vb: VarBuilder) -> Result<Self> {
        let conv = conv1d(
            channels,
            channels,
            kernel_size,
            Conv1dConfig {
                padding: (kernel_size - 1) / 2,
                ..Default::default()
            },
            vb.clone(),
        )?;
        let norm = layer_norm(channels, 1e-5, vb.pp("layer_norm"))?;
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
