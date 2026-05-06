//! Native Rust Kokoro: text → WAV. Milestone 1 CLI.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device};
use kokoro_tts::model::Kokoro;
use kokoro_tts::phonemizer::{Phonemizer, TwoTierPhonemizer, MILESTONE_TEST_PHONEMES};
use std::path::PathBuf;

const SILENCE_PADDING_SAMPLES: usize = 24_000 * 80 / 1_000;
const MAX_SENTENCE_PHONEMES: usize = 510;

#[derive(Debug)]
struct Args {
    model_dir: PathBuf,
    voice: PathBuf,
    text: Option<String>,
    phonemes: Option<String>,
    out: PathBuf,
    speed: f64,
    verbose: bool,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let mut parsed = Self {
            model_dir: PathBuf::from("models"),
            voice: PathBuf::from("models/voices/af_heart.safetensors"),
            text: None,
            phonemes: None,
            out: PathBuf::from("hello.wav"),
            speed: 1.0,
            verbose: false,
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model-dir" => {
                    parsed.model_dir = PathBuf::from(args.next().context("--model-dir")?)
                }
                "--voice" => parsed.voice = PathBuf::from(args.next().context("--voice")?),
                "--text" => parsed.text = Some(args.next().context("--text")?),
                "--phonemes" => parsed.phonemes = Some(args.next().context("--phonemes")?),
                "--out" => parsed.out = PathBuf::from(args.next().context("--out")?),
                "--speed" => parsed.speed = args.next().context("--speed")?.parse()?,
                "--verbose" => parsed.verbose = true,
                "--help" | "-h" => {
                    println!("usage: cargo run --release --bin speak -- [--model-dir DIR] [--voice PATH] [--text \"...\" | --phonemes \"...\"] [--out FILE] [--speed F] [--verbose]");
                    std::process::exit(0);
                }
                other => bail!("unknown argument {other}"),
            }
        }
        Ok(parsed)
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::try_init().ok();
    let args = Args::parse()?;
    let device = Device::Cpu;

    println!("loading model from {} ...", args.model_dir.display());
    let model = Kokoro::load(&args.model_dir, &device).context("loading Kokoro")?;
    println!("model loaded, running forward ...");

    let t0 = std::time::Instant::now();
    let audio = synthesize(&model, &args, &device)?;
    let dt = t0.elapsed();

    let samples = audio
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let max_abs = samples.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let n_samples = samples.len();
    let duration_s = n_samples as f64 / 24_000.0;
    println!(
        "synthesized {} samples ({:.3}s @ 24 kHz) max_abs={:.4} in {:.3}s ({:.2}x realtime)",
        n_samples,
        duration_s,
        max_abs,
        dt.as_secs_f64(),
        duration_s / dt.as_secs_f64()
    );

    // Normalize-soft if peaks > 1 (some test inputs produce huge outputs)
    let scale = if max_abs > 1.0 { 1.0 / max_abs } else { 1.0 };
    let pcm: Vec<i16> = samples
        .iter()
        .map(|&v| (v * scale * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16)
        .collect();
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&args.out, spec)
        .with_context(|| format!("opening {}", args.out.display()))?;
    for &s in &pcm {
        writer.write_sample(s)?;
    }
    writer.finalize()?;
    println!("wrote {}", args.out.display());
    if scale < 1.0 {
        println!(
            "(soft-normalized by {:.4} because peak {:.3} > 1.0)",
            scale, max_abs
        );
    }
    Ok(())
}

fn synthesize(model: &Kokoro, args: &Args, device: &Device) -> Result<candle_core::Tensor> {
    let phonemes_chunks = match (&args.phonemes, &args.text) {
        (Some(p), _) => vec![p.clone()],
        (None, Some(t)) => TwoTierPhonemizer
            .phonemize_chunks(t)
            .with_context(|| format!("phonemizing {:?}", t))?,
        (None, None) => vec![MILESTONE_TEST_PHONEMES.to_string()],
    };

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

        let ref_s = Kokoro::load_voice(&args.voice, phoneme_count, device)
            .with_context(|| format!("loading voice {}", args.voice.display()))?;
        if args.verbose {
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
            .forward(phonemes, &ref_s, args.speed)
            .with_context(|| format!("forward for chunk {}", idx + 1))?;
        let samples = audio
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if args.verbose {
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

    let all = concat_with_silence(&audio_chunks, SILENCE_PADDING_SAMPLES);
    let n_samples = all.len();
    candle_core::Tensor::from_vec(all, (1, n_samples), device)
        .context("assembling chunked audio tensor")
}

#[cfg(test)]
mod tests {
    use super::{concat_with_silence, SILENCE_PADDING_SAMPLES};

    #[test]
    fn inserts_silence_between_chunks_only() {
        let chunks = vec![vec![1.0f32, 2.0], vec![3.0], vec![4.0, 5.0, 6.0]];
        let out = concat_with_silence(&chunks, 2);
        assert_eq!(out, vec![1.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn sample_count_matches_chunks_plus_padding() {
        let chunks = vec![vec![1.0f32; 3], vec![2.0; 4], vec![3.0; 5]];
        let out = concat_with_silence(&chunks, SILENCE_PADDING_SAMPLES);
        assert_eq!(
            out.len(),
            chunks.iter().map(Vec::len).sum::<usize>() + 2 * SILENCE_PADDING_SAMPLES
        );
    }
}

fn concat_with_silence(chunks: &[Vec<f32>], silence_padding_samples: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for (idx, chunk) in chunks.iter().enumerate() {
        out.extend_from_slice(chunk);
        if idx + 1 < chunks.len() {
            out.extend(std::iter::repeat(0.0).take(silence_padding_samples));
        }
    }
    out
}
