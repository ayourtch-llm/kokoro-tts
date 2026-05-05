#![allow(dead_code)]

use super::bert::{AlbertConfig, HiddenAct};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub hidden_dim: usize,
    pub style_dim: usize,
    pub n_layer: usize,
    pub n_token: usize,
    pub n_mels: usize,
    pub max_dur: usize,
    pub dropout: f64,
    pub text_encoder_kernel_size: usize,
    pub plbert: PlBertConfig,
    pub istftnet: IstftnetConfig,
    pub vocab: HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlBertConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_hidden_layers: usize,
    pub dropout: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IstftnetConfig {
    pub upsample_kernel_sizes: Vec<usize>,
    pub upsample_rates: Vec<usize>,
    pub gen_istft_hop_size: usize,
    pub gen_istft_n_fft: usize,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub resblock_kernel_sizes: Vec<usize>,
    pub upsample_initial_channel: usize,
}

impl Config {
    pub fn load(path: &Path) -> candle_core::Result<Self> {
        let file = std::fs::File::open(path).map_err(candle_core::Error::wrap)?;
        let config: Config = serde_json::from_reader(file).map_err(candle_core::Error::wrap)?;
        Ok(config)
    }

    pub fn to_albert_config(&self) -> AlbertConfig {
        AlbertConfig {
            vocab_size: self.n_token,
            embedding_size: 128,
            hidden_size: self.plbert.hidden_size,
            num_hidden_layers: self.plbert.num_hidden_layers,
            num_attention_heads: self.plbert.num_attention_heads,
            intermediate_size: self.plbert.intermediate_size,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout_prob: self.plbert.dropout,
            attention_probs_dropout_prob: self.plbert.dropout,
            max_position_embeddings: self.plbert.max_position_embeddings,
            type_vocab_size: 1,
            initializer_range: 0.02f64,
            layer_norm_eps: 1e-12f64,
        }
    }
}
