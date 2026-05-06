use crate::model::Kokoro;
use crate::phonemizer::Phonemizer;
use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use std::path::Path;

pub const SILENCE_PADDING_SAMPLES: usize = 24_000 * 80 / 1_000;
pub const MAX_SENTENCE_PHONEMES: usize = 510;

pub fn synthesize_text(
    model: &Kokoro,
    phonemizer: &impl Phonemizer,
    text: &str,
    voice: &Path,
    speed: f64,
    device: &Device,
    verbose: bool,
) -> Result<Vec<f32>> {
    let phonemes_chunks = phonemizer
        .phonemize_chunks(text)
        .with_context(|| format!("phonemizing {:?}", text))?;
    if phonemes_chunks.is_empty() {
        bail!("phonemizer produced no chunks");
    }

    let mut audio_chunks = Vec::new();
    for (idx, phonemes) in phonemes_chunks.iter().enumerate() {
        let phoneme_count = phonemes.chars().count();
        if phoneme_count > MAX_SENTENCE_PHONEMES {
            bail!(
                "sentence {} is too long at {} phonemes; insert punctuation to split it",
                idx + 1,
                phoneme_count
            );
        }

        let ref_s = Kokoro::load_voice(voice, phoneme_count, device)
            .with_context(|| format!("loading voice {}", voice.display()))?;
        if verbose {
            println!(
                "chunk {}/{}: phonemes={} voice_shape={:?}",
                idx + 1,
                phonemes_chunks.len(),
                phoneme_count,
                ref_s.dims()
            );
        }

        let chunk_start = std::time::Instant::now();
        let audio = model
            .forward(phonemes, &ref_s, speed)
            .with_context(|| format!("forward for chunk {}", idx + 1))?;
        let samples = audio
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if verbose {
            let elapsed = chunk_start.elapsed();
            let chunk_duration_s = samples.len() as f64 / 24_000.0;
            println!(
                "chunk {}/{}: {} samples ({:.3}s) in {:.3}s ({:.2}x realtime)",
                idx + 1,
                phonemes_chunks.len(),
                samples.len(),
                chunk_duration_s,
                elapsed.as_secs_f64(),
                chunk_duration_s / elapsed.as_secs_f64()
            );
        }
        audio_chunks.push(samples);
    }

    Ok(concat_with_silence(&audio_chunks, SILENCE_PADDING_SAMPLES))
}

pub fn synthesize_phonemes(
    model: &Kokoro,
    phonemes: &str,
    voice: &Path,
    speed: f64,
    device: &Device,
) -> Result<Vec<f32>> {
    let phoneme_count = phonemes.chars().count();
    if phoneme_count > MAX_SENTENCE_PHONEMES {
        bail!(
            "sentence is too long at {} phonemes; insert punctuation to split it",
            phoneme_count
        );
    }
    let ref_s = Kokoro::load_voice(voice, phoneme_count, device)
        .with_context(|| format!("loading voice {}", voice.display()))?;
    let audio = model.forward(phonemes, &ref_s, speed).context("forward")?;
    audio
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
        .context("extracting samples")
}

pub fn concat_with_silence(chunks: &[Vec<f32>], silence_padding_samples: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for (idx, chunk) in chunks.iter().enumerate() {
        out.extend_from_slice(chunk);
        if idx + 1 < chunks.len() {
            out.extend(std::iter::repeat(0.0).take(silence_padding_samples));
        }
    }
    out
}

pub fn samples_to_tensor(samples: Vec<f32>, device: &Device) -> Result<Tensor> {
    let n_samples = samples.len();
    Tensor::from_vec(samples, (1, n_samples), device).context("assembling chunked audio tensor")
}
