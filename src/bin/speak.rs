//! Native Rust Kokoro: text → WAV. Milestone 1 CLI.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device};
use kokoro_tts::audio::play_samples;
use kokoro_tts::model::Kokoro;
use kokoro_tts::phonemizer::{TwoTierPhonemizer, MILESTONE_TEST_PHONEMES};
use kokoro_tts::synthesis::{
    samples_to_tensor, soft_normalize, synthesize_phonemes, synthesize_text, write_wav,
};
use std::path::PathBuf;

#[derive(Debug)]
struct Args {
    model_dir: PathBuf,
    voice: PathBuf,
    text: Option<String>,
    phonemes: Option<String>,
    out: PathBuf,
    speed: f64,
    verbose: bool,
    play: bool,
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
            play: false,
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
                "--play" => parsed.play = true,
                "--help" | "-h" => {
                    println!("usage: cargo run --release --bin speak -- [--model-dir DIR] [--voice PATH] [--text \"...\" | --phonemes \"...\"] [--out FILE] [--speed F] [--verbose] [--play]");
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
    let (samples, scale) = soft_normalize(&samples);
    let n_samples = samples.len();
    let duration_s = n_samples as f64 / 24_000.0;
    println!(
        "synthesized {} samples ({:.3}s @ 24 kHz) max_abs={:.4} in {:.3}s ({:.2}x realtime)",
        n_samples,
        duration_s,
        if scale > 0.0 { 1.0 / scale } else { 0.0 },
        dt.as_secs_f64(),
        duration_s / dt.as_secs_f64()
    );

    if args.play {
        play_samples(&samples, 24_000).context("playback")?;
    }
    write_wav(&samples, &args.out)?;
    println!("wrote {}", args.out.display());
    if scale < 1.0 {
        println!(
            "(soft-normalized by {:.4} because peak {:.3} > 1.0)",
            scale,
            1.0 / scale
        );
    }
    Ok(())
}

fn synthesize(model: &Kokoro, args: &Args, device: &Device) -> Result<candle_core::Tensor> {
    let phonemizer = TwoTierPhonemizer;
    let samples = match (&args.phonemes, &args.text) {
        (Some(p), _) => synthesize_phonemes(model, p, &args.voice, args.speed, device)?,
        (None, Some(t)) => synthesize_text(
            model,
            &phonemizer,
            t,
            &args.voice,
            args.speed,
            device,
            args.verbose,
        )?,
        (None, None) => synthesize_phonemes(
            model,
            MILESTONE_TEST_PHONEMES,
            &args.voice,
            args.speed,
            device,
        )?,
    };
    samples_to_tensor(samples, device)
}

#[cfg(test)]
mod tests {
    use kokoro_tts::synthesis::{concat_with_silence, SILENCE_PADDING_SAMPLES};

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
