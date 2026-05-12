//! Native Rust Kokoro: text → WAV. Milestone 1 CLI.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device};
use kokoro_tts::audio::play_samples;
use kokoro_tts::default_device;
use kokoro_tts::model::Kokoro;
use kokoro_tts::phonemizer::{TwoTierPhonemizer, MILESTONE_TEST_PHONEMES};
use kokoro_tts::synthesis::resolve_resource_path;
use kokoro_tts::synthesis::{
    samples_to_tensor, soft_normalize, synthesize_phonemes, synthesize_text_opts, write_wav,
    ProgressFn, SynthesisOptions, MAX_SENTENCE_PHONEMES, SILENCE_PADDING_SAMPLES,
};
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug)]
struct Args {
    model_dir: PathBuf,
    voice: PathBuf,
    text: Option<String>,
    phonemes: Option<String>,
    infile: Option<PathBuf>,
    out: PathBuf,
    speed: f64,
    verbose: bool,
    play: bool,
    no_split: bool,
    silence_ms: Option<u64>,
    max_phonemes: Option<usize>,
    device: String,
    vocab: Option<PathBuf>,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let mut parsed = Self {
            model_dir: PathBuf::from("models"),
            voice: PathBuf::from("models/voices/af_heart.safetensors"),
            text: None,
            phonemes: None,
            infile: None,
            out: PathBuf::from("hello.wav"),
            speed: 1.0,
            verbose: false,
            play: false,
            no_split: false,
            silence_ms: None,
            max_phonemes: None,
            device: "auto".to_string(),
            vocab: None,
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model-dir" => {
                    parsed.model_dir = PathBuf::from(args.next().context("--model-dir")?)
                }
                "--voice" => parsed.voice = PathBuf::from(args.next().context("--voice")?),
                "--text" => parsed.text = Some(args.next().context("--text")?),
                "--infile" => parsed.infile = Some(PathBuf::from(args.next().context("--infile")?)),
                "--phonemes" => parsed.phonemes = Some(args.next().context("--phonemes")?),
                "--out" => parsed.out = PathBuf::from(args.next().context("--out")?),
                "--speed" => parsed.speed = args.next().context("--speed")?.parse()?,
                "--verbose" => parsed.verbose = true,
                "--play" => parsed.play = true,
                "--no-split" => parsed.no_split = true,
                "--silence-ms" => {
                    parsed.silence_ms = Some(args.next().context("--silence-ms")?.parse::<u64>()?);
                }
                "--max-phonemes" => {
                    parsed.max_phonemes =
                        Some(args.next().context("--max-phonemes")?.parse::<usize>()?);
                }
                "--device" => parsed.device = args.next().context("--device")?,
                "--vocab" => parsed.vocab = Some(PathBuf::from(args.next().context("--vocab")?)),
                "--help" | "-h" => {
                    println!(
                        "usage: cargo run --release --bin speak -- [--model-dir DIR] [--voice PATH]\n\
                         \t[--text \"...\" | --infile FILE | --phonemes \"...\"]\n\
                         \t[--out FILE] [--speed F] [--device auto|cpu|metal] [--verbose] [--play]\n\
                         \t[--no-split] [--silence-ms N] [--max-phonemes N] [--vocab FILE.json]"
                    );
                    std::process::exit(0);
                }
                other => bail!("unknown argument {other}"),
            }
        }
        Ok(parsed)
    }
}

fn resolve_device(spec: &str) -> Result<Device> {
    match spec {
        "auto" => Ok(default_device()),
        "cpu" => Ok(Device::Cpu),
        "metal" => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0).context("Metal device not available")
            }
            #[cfg(not(feature = "metal"))]
            {
                bail!("--device metal requires building with --features metal")
            }
        }
        other => bail!("unknown --device {other}; expected auto, cpu, or metal"),
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .try_init()
        .ok();
    let mut args = Args::parse()?;
    if let Some(ref path) = args.vocab {
        use kokoro_tts::phonemizer::custom_vocab;
        let vocab = custom_vocab::CustomVocab::load(path)
            .with_context(|| format!("loading custom vocab from {}", path.display()))?;
        custom_vocab::set(vocab)?;
        tracing::info!("loaded custom vocab from {}", path.display());
    }
    if let Some(ref infile) = args.infile {
        args.text = Some(
            fs::read_to_string(infile)
                .with_context(|| format!("reading infile {}", infile.display()))?,
        );
    }
    let device = resolve_device(&args.device)?;
    tracing::info!("device: {:?}", device);
    let model_dir = resolve_resource_path(&args.model_dir);
    let voice = resolve_resource_path(&args.voice);

    println!("loading model from {} ...", model_dir.display());
    let model = Kokoro::load(&model_dir, &device).context("loading Kokoro")?;
    println!("model loaded, running forward ...");

    let t0 = std::time::Instant::now();
    let audio = synthesize(&model, &args, &voice, &device)?;
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

fn synthesize(
    model: &Kokoro,
    args: &Args,
    voice: &PathBuf,
    device: &Device,
) -> Result<candle_core::Tensor> {
    let phonemizer = TwoTierPhonemizer;
    let opts = SynthesisOptions {
        split_sentences: !args.no_split,
        silence_padding_samples: args
            .silence_ms
            .map(|ms| (24_000 * ms / 1_000) as usize)
            .unwrap_or(SILENCE_PADDING_SAMPLES),
        max_sentence_phonemes: args.max_phonemes.unwrap_or(MAX_SENTENCE_PHONEMES),
    };
    let progress: ProgressFn = {
        let verbose = args.verbose;
        Box::new(
            move |sentence: &str, current: usize, total: usize, elapsed: Duration| {
                let pct = current as f64 / total as f64 * 100.0;
                let elapsed_s = elapsed.as_secs_f64();
                let eta = if current > 0 && current < total {
                    let avg = elapsed_s / current as f64;
                    let remaining = (total - current) as f64 * avg;
                    format!("{remaining:.1}s")
                } else {
                    String::new()
                };
                let snippet = if sentence.chars().count() > 60 {
                    let cut: String = sentence.chars().take(60).collect();
                    format!("{cut}...")
                } else {
                    sentence.to_string()
                };
                println!(
                "  {snippet:>65.65}  {current:>3}/{total} ({pct:5.1}%) elapsed={elapsed_s:.1}s ETA={eta}"
            );
                if verbose {
                    println!("    (chunk {current}/{total})");
                }
            },
        )
    };
    let samples = match (&args.phonemes, &args.text) {
        (Some(p), _) => synthesize_phonemes(model, p, voice, args.speed, device)?,
        (None, Some(t)) => synthesize_text_opts(
            model,
            &phonemizer,
            t,
            voice,
            args.speed,
            device,
            args.verbose,
            &opts,
            Some(&progress),
        )?,
        (None, None) => {
            synthesize_phonemes(model, MILESTONE_TEST_PHONEMES, voice, args.speed, device)?
        }
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
