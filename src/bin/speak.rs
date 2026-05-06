//! Native Rust Kokoro: text → WAV. Milestone 1 CLI.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device};
use kokoro_tts::model::Kokoro;
use kokoro_tts::phonemizer::{CmudictPhonemizer, Phonemizer, MILESTONE_TEST_PHONEMES};
use std::path::PathBuf;

#[derive(Debug)]
struct Args {
    model_dir: PathBuf,
    voice: PathBuf,
    text: Option<String>,
    phonemes: Option<String>,
    out: PathBuf,
    speed: f64,
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
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model-dir" => parsed.model_dir = PathBuf::from(args.next().context("--model-dir")?),
                "--voice" => parsed.voice = PathBuf::from(args.next().context("--voice")?),
                "--text" => parsed.text = Some(args.next().context("--text")?),
                "--phonemes" => parsed.phonemes = Some(args.next().context("--phonemes")?),
                "--out" => parsed.out = PathBuf::from(args.next().context("--out")?),
                "--speed" => parsed.speed = args.next().context("--speed")?.parse()?,
                "--help" | "-h" => {
                    println!("usage: cargo run --release --bin speak -- [--model-dir DIR] [--voice PATH] [--text \"...\" | --phonemes \"...\"] [--out FILE] [--speed F]");
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

    let phonemes = match (&args.phonemes, &args.text) {
        (Some(p), _) => p.clone(),
        (None, Some(t)) => CmudictPhonemizer
            .phonemize(t)
            .with_context(|| format!("phonemizing {:?}", t))?,
        (None, None) => MILESTONE_TEST_PHONEMES.to_string(),
    };
    println!("phonemes: {phonemes:?}");

    let phoneme_count = phonemes.chars().count();
    let ref_s = Kokoro::load_voice(&args.voice, phoneme_count, &device)
        .with_context(|| format!("loading voice {}", args.voice.display()))?;
    println!("voice: {} (phoneme_count={}) shape={:?}", args.voice.display(), phoneme_count, ref_s.dims());

    println!("loading model from {} ...", args.model_dir.display());
    let model = Kokoro::load(&args.model_dir, &device).context("loading Kokoro")?;
    println!("model loaded, running forward ...");

    let t0 = std::time::Instant::now();
    let audio = model.forward(&phonemes, &ref_s, args.speed).context("forward")?;
    let dt = t0.elapsed();

    let samples = audio.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
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
        println!("(soft-normalized by {:.4} because peak {:.3} > 1.0)", scale, max_abs);
    }
    Ok(())
}
