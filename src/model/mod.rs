#![allow(dead_code)]

pub mod bert;
pub mod config;
pub mod decoder;
pub mod generator;
pub mod predictor;
pub mod source;
pub mod stft;
pub mod text_encoder;

pub use config::Config;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::path::Path;

use self::bert::CustomAlbert;
use self::decoder::Decoder;
use self::predictor::ProsodyPredictor;
use self::text_encoder::TextEncoder;

pub fn alignment_from_durations(durations: &[i64]) -> Result<Vec<Vec<f32>>> {
    let total_frames = durations.iter().try_fold(0usize, |acc, &duration| {
        let duration = usize::try_from(duration).map_err(|_| {
            candle_core::Error::Msg(format!("negative duration {duration} is invalid"))
        })?;
        acc.checked_add(duration).ok_or_else(|| {
            candle_core::Error::Msg("duration frame count overflowed usize".to_string())
        })
    })?;
    let mut alignment = vec![vec![0.0f32; total_frames]; durations.len()];
    let mut cursor = 0usize;
    for (token_idx, &duration) in durations.iter().enumerate() {
        let duration = usize::try_from(duration).map_err(|_| {
            candle_core::Error::Msg(format!("negative duration {duration} is invalid"))
        })?;
        for frame_idx in cursor..cursor + duration {
            alignment[token_idx][frame_idx] = 1.0;
        }
        cursor += duration;
    }
    Ok(alignment)
}

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
        let decoder = Decoder::load(
            config.hidden_dim,
            config.style_dim,
            &config.istftnet,
            vb.pp("decoder"),
        )?;

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

    /// Load a voice's reference style tensor for a given phoneme count.
    /// Returns `[1, 256]` (split as `s_pred = ref_s[:, 128:]` for the predictor
    /// and `s_dec = ref_s[:, :128]` for the decoder per upstream model.py:104,118).
    pub fn load_voice(path: &Path, phoneme_count: usize, device: &Device) -> Result<Tensor> {
        let tensors = candle_core::safetensors::load(path, device)?;
        let ref_s = tensors.get("ref_s").ok_or_else(|| {
            candle_core::Error::Msg(format!("missing ref_s in {}", path.display()))
        })?;
        let n = ref_s.dim(0)?;
        let idx = phoneme_count.min(n.saturating_sub(1));
        Ok(ref_s.narrow(0, idx, 1)?.squeeze(0)?)
    }

    /// Full forward pass: phonemes + reference style -> audio waveform `[1, N_samples]` @ 24 kHz.
    /// Mirrors upstream model.py:forward_with_tokens (model.py:87-119).
    pub fn forward(&self, phonemes: &str, ref_s: &Tensor, speed: f64) -> Result<Tensor> {
        let input_ids_vec = self.phonemes_to_ids(phonemes);
        if input_ids_vec.is_empty() {
            return Err(candle_core::Error::Msg(
                "No valid phonemes found".to_string(),
            ));
        }
        if input_ids_vec.len() + 2 > self.context_length {
            return Err(candle_core::Error::Msg(format!(
                "Input too long: {} + 2 > {}",
                input_ids_vec.len(),
                self.context_length
            )));
        }

        // [0, *input_ids, 0] with BOS/EOS padding
        let mut ids_with_pad = vec![0i64];
        ids_with_pad.extend(input_ids_vec);
        ids_with_pad.push(0);

        let input_len = ids_with_pad.len();
        let input_ids = Tensor::from_vec(ids_with_pad, (1, input_len), ref_s.device())?;
        let text_mask = Tensor::zeros((1, input_len), DType::U8, ref_s.device())?;
        let attention_mask = Tensor::ones((1, input_len), DType::U8, ref_s.device())?;

        // BERT → bert_encoder → [B, hidden, T]
        let bert_dur = self.bert.forward(&input_ids, &attention_mask)?;
        let d_en = self.bert_encoder.forward(&bert_dur)?;
        let d_en = d_en.transpose(1, 2)?;

        // Split reference style: s_pred for predictor, s_dec for decoder.
        // Per upstream model.py:104,118 — verified against checkpoint.
        let s_pred = ref_s.narrow(1, 128, ref_s.dim(1)? - 128)?;
        let s_dec = ref_s.narrow(1, 0, 128)?;

        // Predictor text features (used twice: for duration logits and for F0/N input).
        let d = self.predictor.text_encode(&d_en, &s_pred, &text_mask)?;
        let duration_logits = self.predictor.duration_from_features(&d, &s_pred)?;
        let duration = candle_nn::ops::sigmoid(&duration_logits)?.sum(candle_core::D::Minus1)?;
        let duration = (duration / speed)?;
        let duration_vals = duration.flatten_all()?.to_vec1::<f32>()?;
        let pred_dur: Vec<i64> = duration_vals
            .iter()
            .map(|&x| x.round().max(1.0) as i64)
            .collect();

        // Alignment matrix [1, T, sum(pred_dur)] via one-hot scatter
        let alignment_2d = alignment_from_durations(&pred_dur)?;
        let total_frames: usize = pred_dur.iter().sum::<i64>() as usize;
        let alignment_flat: Vec<f32> = alignment_2d.into_iter().flatten().collect();
        let alignment =
            Tensor::from_vec(alignment_flat, (1, input_len, total_frames), ref_s.device())?;

        // F0/N from predictor: en = d @ alignment, then fn_train re-concats style internally.
        let en = d.matmul(&alignment)?;
        let (f0_pred, n_pred) = self.predictor.fn_train(&en, &s_pred)?;

        // Standalone text encoder (separate from predictor's text encoder).
        let t_en = self.text_encoder.forward(&input_ids)?;
        let asr = t_en.matmul(&alignment)?;

        // Decoder front-end + Generator → waveform [1, T_audio].
        self.decoder.forward(&asr, &f0_pred, &n_pred, &s_dec)
    }
}
