use crate::model::Kokoro;
use crate::phonemizer::Phonemizer;
use crate::phonemizer::normalize::normalize_urls_with;
use crate::phonemizer::{lexicon, misaki_gold};
use crate::phonemizer::sentence::split_sentences;
use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub const SILENCE_PADDING_SAMPLES: usize = 24_000 * 80 / 1_000;
pub const MAX_SENTENCE_PHONEMES: usize = 510;

/// Progress callback: (original_sentence, chunk_index_1based, total_chunks, elapsed)
pub type ProgressFn = Box<dyn Fn(&str, usize, usize, Duration) + Send>;

/// Per-chunk audio callback: (samples, chunk_index_1based, total_chunks).
/// Use to start playback (or any downstream consumer) before the full
/// synthesis is done — lets callers stream chunk-by-chunk.
pub type OnChunkFn = Box<dyn Fn(&[f32], usize, usize) + Send>;

pub struct SynthesisOptions {
    pub silence_padding_samples: usize,
    pub max_sentence_phonemes: usize,
    pub split_sentences: bool,
    /// Number of processed (post-phonemization) chunks to skip before
    /// synthesizing. Useful for resuming a long run partway through.
    pub skip_chunks: usize,
    /// Maximum number of chunks to synthesize in this run, counting
    /// from `skip_chunks`. `None` means no limit.
    pub n_chunks: Option<usize>,
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            silence_padding_samples: SILENCE_PADDING_SAMPLES,
            max_sentence_phonemes: MAX_SENTENCE_PHONEMES,
            split_sentences: true,
            skip_chunks: 0,
            n_chunks: None,
        }
    }
}

pub fn resolve_resource_path(path: &Path) -> PathBuf {
    if path.exists() {
        return path.to_path_buf();
    }
    if !path.is_absolute() {
        if let Ok(current_dir) = std::env::current_dir() {
            let candidate = current_dir.join(path);
            if candidate.exists() {
                return candidate;
            }
        }
        if let Ok(exe) = std::env::current_exe() {
            for ancestor in exe.ancestors().skip(1) {
                let candidate = ancestor.join(path);
                if candidate.exists() {
                    return candidate;
                }
            }
        }
    }
    path.to_path_buf()
}

pub fn synthesize_text(
    model: &Kokoro,
    phonemizer: &impl Phonemizer,
    text: &str,
    voice: &Path,
    speed: f64,
    device: &Device,
    verbose: bool,
) -> Result<Vec<f32>> {
    synthesize_text_opts(
        model,
        phonemizer,
        text,
        voice,
        speed,
        device,
        verbose,
        &SynthesisOptions::default(),
        None,
    )
}

pub fn synthesize_text_opts(
    model: &Kokoro,
    phonemizer: &impl Phonemizer,
    text: &str,
    voice: &Path,
    speed: f64,
    device: &Device,
    verbose: bool,
    opts: &SynthesisOptions,
    progress: Option<&ProgressFn>,
) -> Result<Vec<f32>> {
    synthesize_text_opts_streaming(
        model, phonemizer, text, voice, speed, device, verbose, opts, progress, None,
    )
}

pub fn synthesize_text_opts_streaming(
    model: &Kokoro,
    phonemizer: &impl Phonemizer,
    text: &str,
    voice: &Path,
    speed: f64,
    device: &Device,
    verbose: bool,
    opts: &SynthesisOptions,
    progress: Option<&ProgressFn>,
    on_chunk: Option<&OnChunkFn>,
) -> Result<Vec<f32>> {
    // URL-normalize before sentence splitting; otherwise the splitter
    // cuts "example.com/path" at the dot.
    let gold = misaki_gold::lexicon();
    let lex = lexicon::lexicon();
    let prepped = normalize_urls_with(text, |w| {
        gold.lookup(w).is_some() || lex.lookup(w).is_some()
    });
    let sentences: Vec<String> = if opts.split_sentences {
        split_sentences(&prepped)
    } else {
        vec![prepped]
    };

    enum Chunk {
        Skipped,
        Pause, // an explicit empty-line break from the source text
        Processed(String, String), // (original_sentence, phonemes)
    }

    let mut chunks: Vec<Chunk> = Vec::with_capacity(sentences.len());
    for sentence in &sentences {
        if sentence.trim().is_empty() {
            chunks.push(Chunk::Pause);
            continue;
        }
        let p = phonemizer
            .phonemize(sentence)
            .with_context(|| format!("phonemizing sentence: {:?}", sentence))?;
        if p.is_empty() {
            eprintln!("  (skipping sentence with no phonemes: {sentence:?})");
            chunks.push(Chunk::Skipped);
        } else {
            chunks.push(Chunk::Processed(sentence.to_string(), p));
        }
    }

    let processed: Vec<&Chunk> = chunks
        .iter()
        .filter(|c| matches!(c, Chunk::Processed(..) | Chunk::Pause))
        .collect();
    if !processed.iter().any(|c| matches!(c, Chunk::Processed(..))) {
        bail!("phonemizer produced no chunks");
    }

    let mut audio_chunks: Vec<Vec<f32>> = Vec::new();
    let total = processed.len();
    let start = opts.skip_chunks.min(total);
    let end = match opts.n_chunks {
        Some(n) => start.saturating_add(n).min(total),
        None => total,
    };
    if start >= total && total > 0 {
        bail!(
            "--skip-chunks ({}) is at or past the chunk count ({}); nothing to render",
            opts.skip_chunks,
            total
        );
    }
    if end == start {
        bail!("--n-chunks resolves to 0; nothing to render");
    }
    let window = end - start;
    let t0 = std::time::Instant::now();
    for window_idx in 0..window {
        let idx = start + window_idx;
        let last_in_window = window_idx + 1 == window;
        let (sentence, phonemes) = match &processed[idx] {
            Chunk::Processed(s, p) => (s, p),
            Chunk::Pause => {
                let silence = vec![0.0f32; opts.silence_padding_samples];
                if let Some(cb) = on_chunk {
                    cb(&silence, idx + 1, total);
                }
                audio_chunks.push(silence);
                continue;
            }
            _ => unreachable!(),
        };
        let phoneme_count = phonemes.chars().count();
        if phoneme_count > opts.max_sentence_phonemes {
            bail!(
                "sentence {} is too long at {} phonemes (max {}); \
                 insert punctuation to split it or increase --max-phonemes.\n  \
                 sentence: {:?}",
                idx + 1,
                phoneme_count,
                opts.max_sentence_phonemes,
                sentence,
            );
        }

        let ref_s = Kokoro::load_voice(voice, phoneme_count, device)
            .with_context(|| format!("loading voice {}", voice.display()))?;
        if verbose {
            println!(
                "chunk {}/{}: phonemes={} voice_shape={:?}",
                idx + 1,
                total,
                phoneme_count,
                ref_s.dims()
            );
        }

        let chunk_start = std::time::Instant::now();
        let audio = model.forward(phonemes, &ref_s, speed).with_context(|| {
            format!(
                "forward for chunk {} ({} phonemes)\n  sentence: {:?}",
                idx + 1,
                phoneme_count,
                sentence,
            )
        })?;
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
                total,
                samples.len(),
                chunk_duration_s,
                elapsed.as_secs_f64(),
                chunk_duration_s / elapsed.as_secs_f64()
            );
        }
        if let Some(cb) = on_chunk {
            cb(&samples, idx + 1, total);
            if !last_in_window && opts.silence_padding_samples > 0 {
                let pad = vec![0.0f32; opts.silence_padding_samples];
                cb(&pad, idx + 1, total);
            }
        }
        audio_chunks.push(samples);

        if let Some(cb) = progress {
            cb(sentence, idx + 1, total, t0.elapsed());
        }
    }

    Ok(concat_with_silence(&audio_chunks, opts.silence_padding_samples))
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

pub fn soft_normalize(samples: &[f32]) -> (Vec<f32>, f32) {
    let max_abs = samples.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let scale = if max_abs > 1.0 { 1.0 / max_abs } else { 1.0 };
    let normalized = samples.iter().map(|&v| v * scale).collect();
    (normalized, scale)
}

/// Picks a writer by file extension: `.mp3` → LAME MP3, anything else → WAV.
pub fn write_audio(samples: &[f32], path: &Path) -> Result<()> {
    let is_mp3 = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("mp3"))
        .unwrap_or(false);
    if is_mp3 {
        write_mp3(samples, path)
    } else {
        write_wav(samples, path)
    }
}

pub fn write_mp3(samples: &[f32], path: &Path) -> Result<()> {
    use mp3lame_encoder::{Bitrate, Builder, FlushNoGap, MonoPcm, Quality};

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }

    let mut builder = Builder::new().context("creating LAME builder")?;
    builder
        .set_num_channels(1)
        .map_err(|e| anyhow::anyhow!("LAME set channels: {e}"))?;
    builder
        .set_sample_rate(24_000)
        .map_err(|e| anyhow::anyhow!("LAME set sample rate: {e}"))?;
    builder
        .set_brate(Bitrate::Kbps96)
        .map_err(|e| anyhow::anyhow!("LAME set bitrate: {e}"))?;
    builder
        .set_quality(Quality::Best)
        .map_err(|e| anyhow::anyhow!("LAME set quality: {e}"))?;
    let mut encoder = builder
        .build()
        .map_err(|e| anyhow::anyhow!("initializing LAME encoder: {e}"))?;

    let mut out = Vec::with_capacity(mp3lame_encoder::max_required_buffer_size(samples.len()));
    encoder
        .encode_to_vec(MonoPcm(samples), &mut out)
        .map_err(|e| anyhow::anyhow!("LAME encode: {e}"))?;
    // Reserve room for the final frame(s) before flushing.
    out.reserve(7200);
    encoder
        .flush_to_vec::<FlushNoGap>(&mut out)
        .map_err(|e| anyhow::anyhow!("LAME flush: {e}"))?;

    fs::write(path, &out).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

pub fn write_wav(samples: &[f32], path: &Path) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("opening {}", path.display()))?;
    for &sample in samples {
        let pcm = (sample * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        writer.write_sample(pcm)?;
    }
    writer.finalize()?;
    Ok(())
}

pub fn timestamped_wav_name(now: SystemTime) -> String {
    let duration = now.duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = duration.as_secs() as i64;
    let micros = duration.subsec_micros();
    let days = secs.div_euclid(86_400);
    let secs_of_day = secs.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = secs_of_day / 3_600;
    let minute = (secs_of_day % 3_600) / 60;
    let second = secs_of_day % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}-{minute:02}-{second:02}.{micros:06}Z.wav")
}

fn civil_from_days(days: i64) -> (i32, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    let year = y + if m <= 2 { 1 } else { 0 };
    (year as i32, m as u32, d as u32)
}
