//! Round-trip TTS→ASR accuracy harness.
//!
//! Loads kokoro-tts and nemotron-speech models once, then for each input
//! sentence: synthesizes it, resamples 24k→16k, runs through nemotron's
//! greedy RNN-T decoder, and prints `idx | original | asr | WER`.
//!
//! Build with:
//!     cargo build --release --features accuracy --bin tts_asr_roundtrip
//!
//! Usage:
//!     ./target/release/tts_asr_roundtrip \
//!         --samples tmp/samples.txt \
//!         --start 0 --end 50 \
//!         [--lines 5,10-20,42] \
//!         [--model-dir models] [--voice models/voices/af_heart.safetensors] \
//!         [--nemo-st PATH] [--nemo-tok PATH] [--device cpu|metal|auto]

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use kokoro_tts::default_device;
use kokoro_tts::model::Kokoro;
use kokoro_tts::phonemizer::{Phonemizer, TwoTierPhonemizer};
use kokoro_tts::synthesis::resolve_resource_path;
use nemotron_speech::features::{MelConfig, MelExtractor};
use nemotron_speech::model::ModelConfig as NemoModelConfig;
use nemotron_speech::model::encoder::FastConformerEncoder;
use nemotron_speech::model::greedy::{GreedyDecoder, GreedyDecoderConfig};
use nemotron_speech::model::joint::JointNet;
use nemotron_speech::model::predict::PredictNet;
use nemotron_speech::tokenizer::Tokenizer;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

struct Args {
    samples: PathBuf,
    start: usize,
    end: Option<usize>,
    lines: Option<Vec<(usize, usize)>>, // explicit ranges (inclusive)
    model_dir: PathBuf,
    voice: PathBuf,
    nemo_st: PathBuf,
    nemo_tok: PathBuf,
    device: String,
    speed: f64,
    summary: bool,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut a = Self {
            samples: PathBuf::from("tmp/samples.txt"),
            start: 0,
            end: None,
            lines: None,
            model_dir: PathBuf::from("models"),
            voice: PathBuf::from("models/voices/af_heart.safetensors"),
            nemo_st: PathBuf::from(
                "../nemotron-speech/models/nemotron-speech-streaming-en-0.6b.safetensors",
            ),
            nemo_tok: PathBuf::from("../nemotron-speech/models/tokenizer.model"),
            device: "auto".to_string(),
            speed: 1.0,
            summary: true,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--samples" => a.samples = PathBuf::from(it.next().context("--samples")?),
                "--start" => a.start = it.next().context("--start")?.parse()?,
                "--end" => a.end = Some(it.next().context("--end")?.parse()?),
                "--lines" => {
                    let spec = it.next().context("--lines")?;
                    a.lines = Some(parse_line_spec(&spec)?);
                }
                "--model-dir" => a.model_dir = PathBuf::from(it.next().context("--model-dir")?),
                "--voice" => a.voice = PathBuf::from(it.next().context("--voice")?),
                "--nemo-st" => a.nemo_st = PathBuf::from(it.next().context("--nemo-st")?),
                "--nemo-tok" => a.nemo_tok = PathBuf::from(it.next().context("--nemo-tok")?),
                "--device" => a.device = it.next().context("--device")?,
                "--speed" => a.speed = it.next().context("--speed")?.parse()?,
                "--no-summary" => a.summary = false,
                "-h" | "--help" => {
                    println!(
                        "usage: tts_asr_roundtrip --samples FILE [--start N --end M | --lines spec]\n\
                         optional: --model-dir DIR --voice PATH --nemo-st PATH --nemo-tok PATH\n\
                                   --device cpu|metal|auto --speed F --no-summary\n\
                         --lines spec: comma-separated ranges, e.g. 5,10-20,42"
                    );
                    std::process::exit(0);
                }
                other => bail!("unknown arg {other}"),
            }
        }
        Ok(a)
    }
}

fn parse_line_spec(spec: &str) -> Result<Vec<(usize, usize)>> {
    let mut out = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((lo, hi)) = part.split_once('-') {
            let lo: usize = lo.parse().with_context(|| format!("range lo {part}"))?;
            let hi: usize = hi.parse().with_context(|| format!("range hi {part}"))?;
            if hi < lo {
                bail!("range {part}: hi < lo");
            }
            out.push((lo, hi));
        } else {
            let n: usize = part.parse().with_context(|| format!("index {part}"))?;
            out.push((n, n));
        }
    }
    Ok(out)
}

fn resolve_device(spec: &str) -> Result<Device> {
    match spec {
        "auto" => Ok(default_device()),
        "cpu" => Ok(Device::Cpu),
        "metal" => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0).context("Metal device unavailable")
            }
            #[cfg(not(feature = "metal"))]
            {
                bail!("--device metal requires building with --features metal")
            }
        }
        other => bail!("unknown device {other}; expected auto|cpu|metal"),
    }
}

/// Linear resample 24 kHz → 16 kHz mono f32. Speech is well-bounded below
/// 8 kHz Nyquist so linear interp is adequate for ASR input; we're not
/// chasing perceptual audio quality here.
fn resample_24k_to_16k(input: &[f32]) -> Vec<f32> {
    let out_len = input.len() * 16_000 / 24_000;
    if out_len == 0 || input.len() < 2 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(out_len);
    let ratio = input.len() as f64 / out_len as f64;
    for i in 0..out_len {
        let pos = i as f64 * ratio;
        let t0 = pos.floor() as usize;
        let t1 = (t0 + 1).min(input.len() - 1);
        let frac = (pos - t0 as f64) as f32;
        out.push(input[t0] * (1.0 - frac) + input[t1] * frac);
    }
    out
}

/// Word-level error rate after lowercase + strip punctuation + collapse
/// whitespace. `wer = edits / max(ref_words, hyp_words)`.
fn normalize_for_wer(s: &str) -> Vec<String> {
    s.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

fn word_edit_distance(a: &[String], b: &[String]) -> usize {
    let m = a.len();
    let n = b.len();
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (curr[j - 1] + 1)
                .min(prev[j] + 1)
                .min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

fn wer(reference: &str, hypothesis: &str) -> (f64, usize, usize) {
    let r = normalize_for_wer(reference);
    let h = normalize_for_wer(hypothesis);
    let denom = r.len().max(h.len());
    if denom == 0 {
        return (0.0, 0, 0);
    }
    let d = word_edit_distance(&r, &h);
    (d as f64 / denom as f64, d, denom)
}

fn read_samples(path: &Path) -> Result<Vec<String>> {
    let f = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let r = BufReader::new(f);
    let mut out = Vec::new();
    for line in r.lines() {
        out.push(line?);
    }
    Ok(out)
}

fn selected_indices(args: &Args, total: usize) -> Vec<usize> {
    let mut out = Vec::new();
    if let Some(ranges) = &args.lines {
        for (lo, hi) in ranges {
            for i in *lo..=*hi {
                if i < total {
                    out.push(i);
                }
            }
        }
    } else {
        let end = args.end.unwrap_or(total).min(total);
        for i in args.start..end {
            out.push(i);
        }
    }
    out.sort();
    out.dedup();
    out
}

struct AsrPipeline {
    encoder: FastConformerEncoder,
    predict: PredictNet,
    joint: JointNet,
    decoder: GreedyDecoder,
    tokenizer: Tokenizer,
    mel: MelExtractor,
    cfg: NemoModelConfig,
    device: Device,
    dtype: DType,
}

impl AsrPipeline {
    fn load(st: &Path, tok: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let mel_cfg = MelConfig::nemotron_default();
        let mel = MelExtractor::from_safetensors(st, mel_cfg)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[st.to_path_buf()], dtype, device)? };
        let cfg = NemoModelConfig::nemotron_06b();
        let encoder = FastConformerEncoder::new(vb.pp("encoder"), cfg.clone())
            .map_err(|e| anyhow::anyhow!("encoder: {e:#}"))?;
        let predict =
            PredictNet::new(vb.pp("predict"), &cfg).map_err(|e| anyhow::anyhow!("predict: {e:#}"))?;
        let joint = JointNet::new(vb.pp("joint"), &cfg)
            .map_err(|e| anyhow::anyhow!("joint: {e:#}"))?;
        let decoder = GreedyDecoder::new(
            &predict,
            GreedyDecoderConfig {
                blank_idx: cfg.blank_idx,
                max_symbols_per_step: 10,
            },
            device,
            dtype,
        )?;
        let tokenizer = Tokenizer::from_file(tok)?;
        Ok(Self {
            encoder,
            predict,
            joint,
            decoder,
            tokenizer,
            mel,
            cfg,
            device: device.clone(),
            dtype,
        })
    }

    fn transcribe(&mut self, audio_16k: &[f32]) -> Result<String> {
        let mel_cfg = self.mel.config().clone();
        let log_mel = self.mel.forward(audio_16k);
        let n_frames = log_mel.len() / mel_cfg.n_mels;
        let mel_t = Tensor::from_vec(log_mel, (1, mel_cfg.n_mels, n_frames), &self.device)?
            .to_dtype(self.dtype)?;
        let enc_out = self
            .encoder
            .forward_full(&mel_t, false)
            .map_err(|e| anyhow::anyhow!("encoder forward: {e:#}"))?;
        let enc_seq = enc_out.squeeze(0)?;
        let mut tokens: Vec<u32> = Vec::new();
        self.decoder
            .decode(&enc_seq, &self.predict, &self.joint, &mut tokens)?;
        let text = self.tokenizer.detokenize(&tokens)?;
        Ok(text)
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let args = Args::parse()?;
    let device = resolve_device(&args.device)?;
    let dtype = DType::F32;
    eprintln!("device: {:?}", device);

    // Kokoro
    eprintln!("loading kokoro from {} ...", args.model_dir.display());
    let kokoro = Kokoro::load(&resolve_resource_path(&args.model_dir), &device)
        .map_err(|e| anyhow::anyhow!("kokoro load: {e:#}"))?;
    let voice_path = resolve_resource_path(&args.voice);
    let phonemizer = TwoTierPhonemizer;

    // Nemotron
    let nemo_st = resolve_resource_path(&args.nemo_st);
    let nemo_tok = resolve_resource_path(&args.nemo_tok);
    eprintln!("loading nemotron from {} ...", nemo_st.display());
    let mut asr = AsrPipeline::load(&nemo_st, &nemo_tok, &device, dtype)?;
    let _ = asr.cfg.blank_idx;

    let samples = read_samples(&args.samples)?;
    let indices = selected_indices(&args, samples.len());
    eprintln!(
        "samples loaded: {} total; processing {} ({:?}..{:?})",
        samples.len(),
        indices.len(),
        indices.first(),
        indices.last(),
    );

    let mut total_edits = 0usize;
    let mut total_words = 0usize;
    let mut empty_phon = 0usize;
    let mut errors = 0usize;

    for idx in indices {
        let text = &samples[idx];
        if text.trim().is_empty() {
            empty_phon += 1;
            continue;
        }
        let phonemes = match phonemizer.phonemize(text) {
            Ok(p) => p,
            Err(e) => {
                println!("{idx:>6} | ERR phonemize: {e}");
                println!("       | orig: {text}");
                errors += 1;
                continue;
            }
        };
        if phonemes.is_empty() {
            println!("{idx:>6} | <empty phonemes>");
            println!("       | orig: {text}");
            empty_phon += 1;
            continue;
        }
        let audio_24k = match kokoro_tts::synthesis::synthesize_phonemes(
            &kokoro,
            &phonemes,
            &voice_path,
            args.speed,
            &device,
        ) {
            Ok(a) => a,
            Err(e) => {
                println!("{idx:>6} | ERR synth: {e}");
                println!("       | orig: {text}");
                errors += 1;
                continue;
            }
        };
        let audio_16k = resample_24k_to_16k(&audio_24k);
        if audio_16k.is_empty() {
            println!("{idx:>6} | ERR empty audio after resample");
            errors += 1;
            continue;
        }
        let hyp = match asr.transcribe(&audio_16k) {
            Ok(t) => t,
            Err(e) => {
                println!("{idx:>6} | ERR asr: {e}");
                println!("       | orig: {text}");
                errors += 1;
                continue;
            }
        };
        let (rate, d, denom) = wer(text, &hyp);
        total_edits += d;
        total_words += denom;
        println!(
            "{idx:>6} | WER={:.3} ({}/{} words)",
            rate, d, denom,
        );
        println!("       | orig: {text}");
        println!("       | asr : {}", hyp.trim());
    }

    if args.summary {
        let aggregate = if total_words == 0 {
            0.0
        } else {
            total_edits as f64 / total_words as f64
        };
        println!();
        println!(
            "summary: aggregate WER={:.3} ({} edits / {} ref-words)  empty_phon={} errors={}",
            aggregate, total_edits, total_words, empty_phon, errors,
        );
    }
    Ok(())
}
