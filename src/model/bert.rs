#![allow(dead_code)]

use candle_core::{DType, Module, Result, Tensor, D};
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, VarBuilder};

#[derive(Debug, Clone, Copy)]
pub enum HiddenAct {
    Gelu,
}

#[derive(Debug, Clone)]
pub struct AlbertConfig {
    pub vocab_size: usize,
    pub embedding_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: HiddenAct,
    pub hidden_dropout_prob: f64,
    pub attention_probs_dropout_prob: f64,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
}

#[derive(Clone)]
struct Dropout {
    #[allow(dead_code)]
    p: f64,
}

impl Dropout {
    fn new(p: f64) -> Self {
        Self { p }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _ = self.p;
        Ok(x.clone())
    }
}

struct AlbertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl AlbertEmbeddings {
    fn load(vb: VarBuilder, config: &AlbertConfig) -> Result<Self> {
        let word_embeddings = embedding(
            config.vocab_size,
            config.embedding_size,
            vb.pp("word_embeddings"),
        )?;
        let position_embeddings = embedding(
            config.max_position_embeddings,
            config.embedding_size,
            vb.pp("position_embeddings"),
        )?;
        let token_type_embeddings = embedding(
            config.type_vocab_size,
            config.embedding_size,
            vb.pp("token_type_embeddings"),
        )?;
        let layer_norm = layer_norm(
            config.embedding_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout: Dropout::new(config.hidden_dropout_prob),
        })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;
        let token_type_ids = Tensor::zeros((batch_size, seq_len), DType::U32, input_ids.device())?;
        let position_ids = Tensor::arange(0u32, seq_len as u32, input_ids.device())?;

        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let token_type_embeddings = self.token_type_embeddings.forward(&token_type_ids)?;
        let position_embeddings = self.position_embeddings.forward(&position_ids)?;

        let embeddings = (input_embeddings + token_type_embeddings)?
            .broadcast_add(&position_embeddings)?
            .apply(&self.layer_norm)?;
        self.dropout.forward(&embeddings)
    }
}

struct AlbertAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dense: Linear,
    layer_norm: LayerNorm,
    attention_dropout: Dropout,
    output_dropout: Dropout,
    num_attention_heads: usize,
    attention_head_size: usize,
}

impl AlbertAttention {
    fn load(vb: VarBuilder, config: &AlbertConfig) -> Result<Self> {
        if config.hidden_size % config.num_attention_heads != 0 {
            candle_core::bail!(
                "ALBERT hidden_size {} is not divisible by num_attention_heads {}",
                config.hidden_size,
                config.num_attention_heads
            );
        }
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        Ok(Self {
            query: linear(config.hidden_size, all_head_size, vb.pp("query"))?,
            key: linear(config.hidden_size, all_head_size, vb.pp("key"))?,
            value: linear(config.hidden_size, all_head_size, vb.pp("value"))?,
            dense: linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?,
            layer_norm: layer_norm(
                config.hidden_size,
                config.layer_norm_eps,
                vb.pp("LayerNorm"),
            )?,
            attention_dropout: Dropout::new(config.attention_probs_dropout_prob),
            output_dropout: Dropout::new(config.hidden_dropout_prob),
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
        })
    }

    fn transpose_for_scores(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        x.reshape((
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        ))?
        .transpose(1, 2)?
        .contiguous()
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let query_layer = self.transpose_for_scores(&self.query.forward(hidden_states)?)?;
        let key_layer = self.transpose_for_scores(&self.key.forward(hidden_states)?)?;
        let value_layer = self.transpose_for_scores(&self.value.forward(hidden_states)?)?;

        let attention_scores = query_layer.matmul(&key_layer.t()?)?;
        let attention_scores = (attention_scores / (self.attention_head_size as f64).sqrt())?;
        let attention_scores = attention_scores.broadcast_add(attention_mask)?;
        let attention_probs = candle_nn::ops::softmax(&attention_scores, D::Minus1)?;
        let attention_probs = self.attention_dropout.forward(&attention_probs)?;

        let context = attention_probs.matmul(&value_layer)?;
        let context = context.transpose(1, 2)?.contiguous()?;
        let (batch_size, seq_len, _, _) = context.dims4()?;
        let context = context.reshape((
            batch_size,
            seq_len,
            self.num_attention_heads * self.attention_head_size,
        ))?;
        let attention_output = self.dense.forward(&context)?;
        let attention_output = self.output_dropout.forward(&attention_output)?;
        self.layer_norm
            .forward(&(hidden_states + attention_output)?)
    }
}

struct AlbertLayer {
    attention: AlbertAttention,
    ffn: Linear,
    ffn_output: Linear,
    full_layer_layer_norm: LayerNorm,
    dropout: Dropout,
    hidden_act: HiddenAct,
}

impl AlbertLayer {
    fn load(vb: VarBuilder, config: &AlbertConfig) -> Result<Self> {
        Ok(Self {
            attention: AlbertAttention::load(vb.pp("attention"), config)?,
            ffn: linear(config.hidden_size, config.intermediate_size, vb.pp("ffn"))?,
            ffn_output: linear(
                config.intermediate_size,
                config.hidden_size,
                vb.pp("ffn_output"),
            )?,
            full_layer_layer_norm: layer_norm(
                config.hidden_size,
                config.layer_norm_eps,
                vb.pp("full_layer_layer_norm"),
            )?,
            dropout: Dropout::new(config.hidden_dropout_prob),
            hidden_act: config.hidden_act,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let attention_output = self.attention.forward(hidden_states, attention_mask)?;
        let ffn_output = self.ffn.forward(&attention_output)?;
        let ffn_output = match self.hidden_act {
            HiddenAct::Gelu => ffn_output.gelu_erf()?,
        };
        let ffn_output = self.ffn_output.forward(&ffn_output)?;
        let ffn_output = self.dropout.forward(&ffn_output)?;
        self.full_layer_layer_norm
            .forward(&(ffn_output + attention_output)?)
    }
}

struct AlbertEncoder {
    embedding_hidden_mapping_in: Linear,
    shared_layer: AlbertLayer,
    num_hidden_layers: usize,
}

impl AlbertEncoder {
    fn load(vb: VarBuilder, config: &AlbertConfig) -> Result<Self> {
        let embedding_hidden_mapping_in = linear(
            config.embedding_size,
            config.hidden_size,
            vb.pp("embedding_hidden_mapping_in"),
        )?;
        let shared_layer =
            AlbertLayer::load(vb.pp("albert_layer_groups.0.albert_layers.0"), config)?;
        Ok(Self {
            embedding_hidden_mapping_in,
            shared_layer,
            num_hidden_layers: config.num_hidden_layers,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let mut hidden_states = self.embedding_hidden_mapping_in.forward(hidden_states)?;
        for _ in 0..self.num_hidden_layers {
            hidden_states = self.shared_layer.forward(&hidden_states, attention_mask)?;
        }
        Ok(hidden_states)
    }
}

/// Custom ALBERT wrapper for PL-BERT. Returns last_hidden_state [B, T, H].
pub struct CustomAlbert {
    embeddings: AlbertEmbeddings,
    encoder: AlbertEncoder,
}

impl CustomAlbert {
    pub fn load(vb: VarBuilder, config: AlbertConfig) -> Result<Self> {
        Ok(Self {
            embeddings: AlbertEmbeddings::load(vb.pp("embeddings"), &config)?,
            encoder: AlbertEncoder::load(vb.pp("encoder"), &config)?,
        })
    }

    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let embedding_output = self.embeddings.forward(input_ids)?;
        let attention_mask = get_extended_attention_mask(attention_mask)?;
        self.encoder.forward(&embedding_output, &attention_mask)
    }
}

fn get_extended_attention_mask(attention_mask: &Tensor) -> Result<Tensor> {
    let attention_mask = match attention_mask.rank() {
        3 => attention_mask.unsqueeze(1)?,
        2 => attention_mask.unsqueeze(1)?.unsqueeze(1)?,
        _ => candle_core::bail!("attention_mask must have rank 2 or 3"),
    };
    let attention_mask = attention_mask.to_dtype(DType::F32)?;
    (attention_mask.ones_like()? - &attention_mask)?
        .broadcast_mul(&Tensor::new(f32::MIN, attention_mask.device())?)
}

#[cfg(test)]
mod tests {
    use super::{AlbertConfig, CustomAlbert, HiddenAct};
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn custom_albert_forward_returns_last_hidden_state() {
        let config = AlbertConfig {
            vocab_size: 8,
            embedding_size: 4,
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            intermediate_size: 16,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            max_position_embeddings: 16,
            type_vocab_size: 1,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
        };
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = CustomAlbert::load(vb, config).unwrap();
        let input_ids = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &device).unwrap();
        let attention_mask = Tensor::ones((1, 3), DType::U8, &device).unwrap();

        let output = model.forward(&input_ids, &attention_mask).unwrap();

        assert_eq!(output.dims(), &[1, 3, 8]);
    }
}
