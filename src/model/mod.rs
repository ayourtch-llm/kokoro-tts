#![allow(dead_code)]

pub mod bert;
mod config;
mod decoder;
mod predictor;
mod text_encoder;

pub use config::Config;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::path::Path;

use self::bert::CustomAlbert;
use self::decoder::Decoder;
use self::predictor::ProsodyPredictor;
use self::text_encoder::TextEncoder;

/// Full Kokoro model
pub struct Kokoro {
    bert: CustomAlbert,
    bert_encoder: candle_nn::Linear,
    predictor: ProsodyPredictor,
    text_encoder: TextEncoder,
    decoder: Decoder,
    vocab: HashMap<String, usize>,
    context_length: usize,
    config: Config,
}

impl Kokoro {
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let config = Config::load(path.join("config.json").as_path())?;
        let albert_config = config.to_albert_config();

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[path.join("model.safetensors")],
                DType::F32,
                device,
            )?
        };

        let bert = CustomAlbert::load(vb.pp("bert"), albert_config.clone())?;
        let bert_encoder = candle_nn::linear(
            config.plbert.hidden_size,
            config.hidden_dim,
            vb.pp("bert_encoder"),
        )?;
        let predictor = ProsodyPredictor::load(
            config.style_dim,
            config.hidden_dim,
            config.n_layer,
            config.max_dur,
            config.dropout,
            vb.pp("predictor"),
        )?;
        let text_encoder = TextEncoder::load(
            config.hidden_dim,
            config.text_encoder_kernel_size,
            config.n_layer,
            config.n_token,
            vb.pp("text_encoder"),
        )?;
        let decoder = Decoder::load(config.hidden_dim, config.style_dim, vb.pp("decoder"))?;

        Ok(Self {
            bert,
            bert_encoder,
            predictor,
            text_encoder,
            decoder,
            vocab: config.vocab.clone(),
            context_length: albert_config.max_position_embeddings,
            config,
        })
    }

    /// Convert phoneme string to input IDs
    pub fn phonemes_to_ids(&self, phonemes: &str) -> Vec<i64> {
        phonemes
            .chars()
            .filter_map(|p| {
                let key = p.to_string();
                match self.vocab.get(&key) {
                    Some(id) => Some(*id as i64),
                    None => {
                        tracing::warn!(phoneme = %p, "dropping unmapped phoneme");
                        None
                    }
                }
            })
            .collect()
    }

    /// Full forward pass: phonemes + reference style -> audio
    pub fn forward(&self, phonemes: &str, ref_s: &Tensor, speed: f64) -> Result<Tensor> {
        let input_ids = self.phonemes_to_ids(phonemes);
        if input_ids.is_empty() {
            return Err(candle_core::Error::Msg(
                "No valid phonemes found".to_string(),
            ));
        }
        if input_ids.len() + 2 > self.context_length {
            return Err(candle_core::Error::Msg(format!(
                "Input too long: {} + 2 > {}",
                input_ids.len(),
                self.context_length
            )));
        }

        // [0, *input_ids, 0] with BOS/EOS padding
        let mut ids_with_pad = vec![0i64];
        ids_with_pad.extend(input_ids);
        ids_with_pad.push(0);

        let input_len = ids_with_pad.len();
        let input_ids = Tensor::from_vec(ids_with_pad, (1, input_len), ref_s.device())?;
        let text_mask = Tensor::zeros((1, input_len), DType::U8, ref_s.device())?;
        let attention_mask = Tensor::ones((1, input_len), DType::U8, ref_s.device())?;

        // BERT encoding
        let bert_dur = self.bert.forward(&input_ids, &attention_mask)?;
        let d_en = self.bert_encoder.forward(&bert_dur)?;
        let d_en = d_en.transpose(1, 2)?; // [B, C, T]

        // Split reference style
        let s = ref_s.narrow(1, 128, ref_s.dim(1)? - 128)?;

        // Duration prediction
        let duration = self.predictor.predict_duration(&d_en, &s, &text_mask)?;
        let duration = candle_nn::ops::sigmoid(&duration)?;
        let duration = duration.sum(candle_core::D::Minus1)?;
        let duration = duration / speed;

        // TODO: Full alignment, F0/N prediction, text encoding, decoding
        // This is where the complex alignment + vocoder pipeline goes

        duration
    }
}
