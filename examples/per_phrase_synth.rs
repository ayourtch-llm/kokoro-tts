//! Minimal library usage: synthesize one phrase at a time and write
//! the audio as a wav. Copy this as a starting point for a downstream
//! project that wants to manage its own playback / queueing /
//! pivot-on-revise logic.
//!
//! Run from inside this repo:
//!   cargo run --release --features metal --example per_phrase_synth
//!
//! The `metal` feature is optional; CPU works too (slower).

use anyhow::Result;
use candle_core::DType;
use kokoro_tts::default_device;
use kokoro_tts::model::Kokoro;
use kokoro_tts::phonemizer::{Phonemizer, TwoTierPhonemizer};
use kokoro_tts::synthesis::{resolve_resource_path, write_wav};
use std::path::Path;

const SAMPLE_RATE: u32 = 24_000;

fn main() -> Result<()> {
    let device = default_device();
    let model_dir = resolve_resource_path(Path::new("models"));
    let voice = resolve_resource_path(Path::new("models/voices/af_heart.safetensors"));
    let kokoro = Kokoro::load(&model_dir, &device)?;
    let phonemizer = TwoTierPhonemizer;

    // Treat each line as a separately-synthesized phrase. A real
    // streaming consumer (LLM-driven, say) would receive these from
    // its token stream, push to its own pivot / speculative queue,
    // and play through whatever audio sink it prefers.
    let phrases = [
        "The first phrase synthesizes on its own.",
        "The second one is queued right after, no concatenation needed.",
        "And the third arrives whenever it does — no shared state between them.",
    ];

    let mut all_samples: Vec<f32> = Vec::new();
    for (i, phrase) in phrases.iter().enumerate() {
        let phonemes = phonemizer.phonemize(phrase)?;
        let phoneme_count = phonemes.chars().count();
        let ref_s = Kokoro::load_voice(&voice, phoneme_count, &device)?;
        let audio = kokoro.forward(&phonemes, &ref_s, 1.0)?;
        let samples: Vec<f32> = audio
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let secs = samples.len() as f64 / SAMPLE_RATE as f64;
        println!("phrase {}: {} phonemes -> {:.2}s of audio", i + 1, phoneme_count, secs);
        // 80 ms of silence between phrases — your project would pick
        // whatever transition fits (crossfade, hard cut, longer pause).
        if i > 0 {
            all_samples.extend(std::iter::repeat(0.0).take((SAMPLE_RATE as usize) * 80 / 1000));
        }
        all_samples.extend(samples);
    }

    let out = Path::new("/tmp/per_phrase_synth.wav");
    write_wav(&all_samples, out)?;
    println!("wrote {}", out.display());
    Ok(())
}
