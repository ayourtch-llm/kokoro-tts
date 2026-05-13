//! Native Rust Kokoro: text → WAV. Milestone 1 CLI.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device};
use kokoro_tts::audio::{play_samples, StreamingAudioOutput};
use kokoro_tts::chapters::split_chapters;
use kokoro_tts::default_device;
use kokoro_tts::model::Kokoro;
use kokoro_tts::phonemizer::{TwoTierPhonemizer, MILESTONE_TEST_PHONEMES};
use kokoro_tts::synthesis::resolve_resource_path;
use kokoro_tts::synthesis::{
    samples_to_tensor, soft_normalize, synthesize_phonemes, write_audio,
    ProgressFn, SynthesisOptions, MAX_SENTENCE_PHONEMES, SILENCE_PADDING_SAMPLES,
};
use std::fs;
use std::path::{Path, PathBuf};
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
    skip_chunks: usize,
    n_chunks: Option<usize>,
    auto_split: bool,
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
            skip_chunks: 0,
            n_chunks: None,
            auto_split: false,
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
                "--skip-chunks" => {
                    parsed.skip_chunks =
                        args.next().context("--skip-chunks")?.parse::<usize>()?;
                }
                "--n-chunks" => {
                    parsed.n_chunks = Some(args.next().context("--n-chunks")?.parse::<usize>()?);
                }
                "--auto-split" => parsed.auto_split = true,
                "--help" | "-h" => {
                    println!(
                        "usage: cargo run --release --bin speak -- [--model-dir DIR] [--voice PATH]\n\
                         \t[--text \"...\" | --infile FILE | --phonemes \"...\"]\n\
                         \t[--out FILE] [--speed F] [--device auto|cpu|metal] [--verbose] [--play]\n\
                         \t[--no-split] [--silence-ms N] [--max-phonemes N] [--vocab FILE.json]\n\
                         \t[--skip-chunks N] [--n-chunks N] [--auto-split]\n\
                         output is WAV by default, or MP3 if --out ends with .mp3\n\
                         with --auto-split, detected chapters are written to separate files\n\
                         named OUT_001.ext, OUT_002.ext, …"
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

    if args.auto_split {
        run_auto_split(&model, &args, &voice, &device)
    } else {
        let t0 = std::time::Instant::now();
        let audio = synthesize(&model, &args, &voice, &device, None)?;
        finalize_and_write(audio, &args.out, t0.elapsed())
    }
}

/// Soft-normalize, log timing, and write the WAV/MP3 at `out_path`.
/// Streaming playback already happened inside `synthesize()` per chunk.
fn finalize_and_write(
    audio: candle_core::Tensor,
    out_path: &Path,
    dt: std::time::Duration,
) -> Result<()> {
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
    let _ = play_samples; // keep the import alive for potential future use
    write_audio(&samples, out_path)?;
    println!("wrote {}", out_path.display());
    if scale < 1.0 {
        println!(
            "(soft-normalized by {:.4} because peak {:.3} > 1.0)",
            scale,
            1.0 / scale
        );
    }
    Ok(())
}

fn run_auto_split(
    model: &Kokoro,
    args: &Args,
    voice: &PathBuf,
    device: &Device,
) -> Result<()> {
    let text = args
        .text
        .as_deref()
        .context("--auto-split requires --text or --infile")?;
    let chapters = split_chapters(text);
    if chapters.is_empty() {
        bail!("no chapters detected (input is empty or whitespace-only)");
    }
    let total = chapters.len();
    println!("auto-split: detected {total} chapter(s)");
    for (idx, chapter_text) in chapters.iter().enumerate() {
        let chapter_num = idx + 1;
        let out_path = chapter_filename(&args.out, chapter_num, total);
        let preview: String = chapter_text
            .lines()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("")
            .chars()
            .take(72)
            .collect();
        println!(
            "\n=== chapter {chapter_num}/{total} → {} ===\n  preview: {preview}",
            out_path.display()
        );
        let t0 = std::time::Instant::now();
        let audio = synthesize(model, args, voice, device, Some(chapter_text))?;
        finalize_and_write(audio, &out_path, t0.elapsed())?;
    }
    Ok(())
}

/// Builds `<stem>_NNN.<ext>` next to `out`, with zero-pad width that
/// fits `total`. So total=12 → `_001`…`_012`; total=2000 → `_0001`.
fn chapter_filename(out: &Path, idx: usize, total: usize) -> PathBuf {
    let width = total.to_string().len().max(3);
    let stem = out
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("chapter");
    let ext = out.extension().and_then(|s| s.to_str());
    let name = match ext {
        Some(e) => format!("{stem}_{idx:0width$}.{e}"),
        None => format!("{stem}_{idx:0width$}"),
    };
    match out.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.join(name),
        _ => PathBuf::from(name),
    }
}

fn synthesize(
    model: &Kokoro,
    args: &Args,
    voice: &PathBuf,
    device: &Device,
    text_override: Option<&str>,
) -> Result<candle_core::Tensor> {
    let phonemizer = TwoTierPhonemizer;
    let opts = SynthesisOptions {
        split_sentences: !args.no_split,
        silence_padding_samples: args
            .silence_ms
            .map(|ms| (24_000 * ms / 1_000) as usize)
            .unwrap_or(SILENCE_PADDING_SAMPLES),
        max_sentence_phonemes: args.max_phonemes.unwrap_or(MAX_SENTENCE_PHONEMES),
        skip_chunks: args.skip_chunks,
        n_chunks: args.n_chunks,
    };
    // If --play, open the audio device up front and enqueue each chunk
    // as it finishes synthesizing — so playback starts after chunk 1
    // rather than after the whole file is rendered.
    let streaming_audio = if args.play {
        Some(StreamingAudioOutput::open().context("opening audio output")?)
    } else {
        None
    };
    let on_chunk: Option<kokoro_tts::synthesis::OnChunkFn> = streaming_audio.as_ref().map(|out| {
        let handle = out.handle();
        // Throttle: synth runs ~3x realtime, so without backpressure
        // the queue blows past the 30s backlog warning quickly. Wait
        // for the buffer to drop below this threshold before sending
        // the next chunk.
        const MAX_BUFFERED_SECONDS: f64 = 8.0;
        let output_rate = out.output_sample_rate() as f64;
        let max_pending = (MAX_BUFFERED_SECONDS * output_rate) as usize;
        Box::new(move |samples: &[f32], _idx: usize, _total: usize| {
            while handle.pending_samples() > max_pending {
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            if let Err(e) = handle.enqueue_samples(samples, 24_000) {
                eprintln!("playback enqueue error: {e:#}");
            }
        }) as kokoro_tts::synthesis::OnChunkFn
    });
    let progress: ProgressFn = {
        let verbose = args.verbose;
        let skip = args.skip_chunks;
        let n_chunks = args.n_chunks;
        Box::new(
            move |sentence: &str, current: usize, total: usize, elapsed: Duration| {
                // Window-relative counters: when --skip-chunks/--n-chunks
                // are in effect, `current` is the absolute index but the
                // ETA must be computed against the slice actually being
                // rendered this run.
                let window_size = match n_chunks {
                    Some(n) => n.min(total.saturating_sub(skip)),
                    None => total.saturating_sub(skip),
                };
                let window_done = current.saturating_sub(skip);
                let pct = if window_size > 0 {
                    window_done as f64 / window_size as f64 * 100.0
                } else {
                    0.0
                };
                let elapsed_s = elapsed.as_secs_f64();
                let eta = if window_done > 0 && window_done < window_size {
                    let avg = elapsed_s / window_done as f64;
                    let remaining = (window_size - window_done) as f64 * avg;
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
                let label = if skip == 0 && n_chunks.is_none() {
                    format!("{current:>3}/{total}")
                } else {
                    format!("{window_done:>3}/{window_size} [src {current}/{total}]")
                };
                println!(
                    "  {snippet:>65.65}  {label} ({pct:5.1}%) elapsed={elapsed_s:.1}s ETA={eta}"
                );
                if verbose {
                    println!("    (chunk {current}/{total})");
                }
            },
        )
    };
    // `text_override` (set by --auto-split for the per-chapter loop)
    // takes precedence over args.text.
    let text = text_override.or(args.text.as_deref());
    let samples = match (&args.phonemes, text) {
        (Some(p), _) => synthesize_phonemes(model, p, voice, args.speed, device)?,
        (None, Some(t)) => kokoro_tts::synthesis::synthesize_text_opts_streaming(
            model,
            &phonemizer,
            t,
            voice,
            args.speed,
            device,
            args.verbose,
            &opts,
            Some(&progress),
            on_chunk.as_ref(),
        )?,
        (None, None) => {
            synthesize_phonemes(model, MILESTONE_TEST_PHONEMES, voice, args.speed, device)?
        }
    };
    // If we were streaming audio, wait for the queue to drain before
    // dropping the output device — otherwise the last few chunks get
    // cut off when speak exits.
    if let Some(out) = streaming_audio {
        out.handle().wait_until_drained();
        drop(out);
    }
    samples_to_tensor(samples, device)
}

#[cfg(test)]
mod tests {
    use super::chapter_filename;
    use kokoro_tts::synthesis::{concat_with_silence, SILENCE_PADDING_SAMPLES};
    use std::path::{Path, PathBuf};

    #[test]
    fn chapter_filename_pads_width_to_three() {
        let out = Path::new("book.mp3");
        assert_eq!(chapter_filename(out, 1, 5), PathBuf::from("book_001.mp3"));
        assert_eq!(chapter_filename(out, 12, 12), PathBuf::from("book_012.mp3"));
    }

    #[test]
    fn chapter_filename_grows_width_for_many_chapters() {
        let out = Path::new("novel.wav");
        assert_eq!(chapter_filename(out, 7, 1500), PathBuf::from("novel_0007.wav"));
    }

    #[test]
    fn chapter_filename_preserves_parent_dir() {
        let out = Path::new("/tmp/audio/book.mp3");
        assert_eq!(
            chapter_filename(out, 3, 9),
            PathBuf::from("/tmp/audio/book_003.mp3")
        );
    }

    #[test]
    fn chapter_filename_handles_no_extension() {
        let out = Path::new("recording");
        assert_eq!(chapter_filename(out, 2, 4), PathBuf::from("recording_002"));
    }

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
