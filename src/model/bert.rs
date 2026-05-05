#![allow(dead_code)]

use candle_core::{Error, Result, Tensor};

use super::config::AlbertConfig;

/// CustomAlbert wrapper - returns last_hidden_state directly
pub struct CustomAlbert {
    _config: AlbertConfig,
}

impl CustomAlbert {
    pub fn load(vb: candle_nn::VarBuilder, config: AlbertConfig) -> Result<Self> {
        let _ = vb;
        Ok(Self { _config: config })
    }

    /// Forward pass returning last_hidden_state [B, T, H]
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let _ = (input_ids, attention_mask);
        Err(Error::Msg(
            "CustomAlbert is not implemented for this candle-transformers version".to_string(),
        ))
    }
}
