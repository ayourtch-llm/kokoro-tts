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

    let total = sentences.len();
    if total == 0 {
        bail!("no sentences to render");
    }
    let start = opts.skip_chunks.min(total);
    let end = match opts.n_chunks {
        Some(n) => start.saturating_add(n).min(total),
        None => total,
    };
    if start >= total {
        bail!(
            "--skip-chunks ({}) is at or past the sentence count ({}); nothing to render",
            opts.skip_chunks,
            total
        );
    }
    if end == start {
        bail!("--n-chunks resolves to 0; nothing to render");
    }
    let window = end - start;

    let mut audio_chunks: Vec<Vec<f32>> = Vec::new();
    let t0 = std::time::Instant::now();
    for window_idx in 0..window {
        let idx = start + window_idx;
        let last_in_window = window_idx + 1 == window;
        let sentence = &sentences[idx];

        if sentence.trim().is_empty() {
            // Explicit empty-line break in the source text: emit silence.
            let silence = vec![0.0f32; opts.silence_padding_samples];
            if let Some(cb) = on_chunk {
                cb(&silence, idx + 1, total);
            }
            audio_chunks.push(silence);
            if let Some(cb) = progress {
                cb(sentence, idx + 1, total, t0.elapsed());
            }
            continue;
        }

        let phoneme_pieces =
            phonemize_with_fallback(sentence, phonemizer, opts.max_sentence_phonemes, idx + 1);
        if phoneme_pieces.is_empty() {
            eprintln!("  (sentence {} produced no usable phonemes; skipping)", idx + 1);
            if let Some(cb) = progress {
                cb(sentence, idx + 1, total, t0.elapsed());
            }
            continue;
        }

        let chunk_start = std::time::Instant::now();
        let mut sentence_samples: Vec<f32> = Vec::new();
        for (piece_idx, phonemes) in phoneme_pieces.iter().enumerate() {
            let phoneme_count = phonemes.chars().count();
            let ref_s = Kokoro::load_voice(voice, phoneme_count, device)
                .with_context(|| format!("loading voice {}", voice.display()))?;
            if verbose {
                println!(
                    "chunk {}/{}{}: phonemes={} voice_shape={:?}",
                    idx + 1,
                    total,
                    if phoneme_pieces.len() > 1 {
                        format!(" piece {}/{}", piece_idx + 1, phoneme_pieces.len())
                    } else {
                        String::new()
                    },
                    phoneme_count,
                    ref_s.dims()
                );
            }
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
            if let Some(cb) = on_chunk {
                cb(&samples, idx + 1, total);
            }
            sentence_samples.extend_from_slice(&samples);
        }
        if let Some(cb) = on_chunk {
            if !last_in_window && opts.silence_padding_samples > 0 {
                let pad = vec![0.0f32; opts.silence_padding_samples];
                cb(&pad, idx + 1, total);
            }
        }
        if verbose {
            let elapsed = chunk_start.elapsed();
            let chunk_duration_s = sentence_samples.len() as f64 / 24_000.0;
            println!(
                "chunk {}/{}: {} samples ({:.3}s) in {:.3}s ({:.2}x realtime)",
                idx + 1,
                total,
                sentence_samples.len(),
                chunk_duration_s,
                elapsed.as_secs_f64(),
                chunk_duration_s / elapsed.as_secs_f64()
            );
        }
        audio_chunks.push(sentence_samples);

        if let Some(cb) = progress {
            cb(sentence, idx + 1, total, t0.elapsed());
        }
    }

    if audio_chunks.is_empty() {
        bail!("nothing rendered: window contained no audible sentences");
    }
    Ok(concat_with_silence(&audio_chunks, opts.silence_padding_samples))
}

/// Phonemize a sentence, returning one or more phoneme strings, each
/// sized to fit `max`. For typical short sentences this returns a
/// single-element vec. For sentences that overflow the model's cap
/// (acks run-ons, index entries, etc.) the splitter falls back through:
///
/// 1. Split the source on commas; phonemize and accept each piece that
///    fits.
/// 2. For any piece still over `max`, halve its word list recursively
///    until each half fits.
/// 3. A fragment that's a single word still over `max` (vanishingly
///    rare in real text) is dropped with a warning.
///
/// Returns an empty vec only when nothing usable came out — caller
/// should skip the sentence in that case.
fn phonemize_with_fallback(
    sentence: &str,
    phonemizer: &impl Phonemizer,
    max: usize,
    sentence_idx: usize,
) -> Vec<String> {
    let phonemes = match phonemizer.phonemize(sentence) {
        Ok(p) => p,
        Err(e) => {
            eprintln!(
                "  (sentence {} phonemize error: {e:#}; skipping)",
                sentence_idx
            );
            return Vec::new();
        }
    };
    if phonemes.is_empty() {
        return Vec::new();
    }
    if phonemes.chars().count() <= max {
        return vec![phonemes];
    }
    eprintln!(
        "  (sentence {} is {} phonemes > {} cap; resplitting on commas/words)",
        sentence_idx,
        phonemes.chars().count(),
        max
    );
    resplit_by_commas(sentence, phonemizer, max)
}

fn resplit_by_commas(text: &str, phonemizer: &impl Phonemizer, max: usize) -> Vec<String> {
    let pieces: Vec<&str> = text
        .split(',')
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .collect();
    if pieces.len() <= 1 {
        // Nothing useful to split on. Skip straight to word-halving.
        return halve_by_words(text, phonemizer, max);
    }
    let mut out = Vec::new();
    for piece in pieces {
        let Ok(phonemes) = phonemizer.phonemize(piece) else {
            continue;
        };
        if phonemes.is_empty() {
            continue;
        }
        if phonemes.chars().count() <= max {
            out.push(phonemes);
        } else {
            out.extend(halve_by_words(piece, phonemizer, max));
        }
    }
    out
}

fn halve_by_words(text: &str, phonemizer: &impl Phonemizer, max: usize) -> Vec<String> {
    let Ok(phonemes) = phonemizer.phonemize(text) else {
        return Vec::new();
    };
    if phonemes.is_empty() {
        return Vec::new();
    }
    if phonemes.chars().count() <= max {
        return vec![phonemes];
    }
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() <= 1 {
        eprintln!(
            "  (dropping un-splittable fragment of {} phonemes: {text:?})",
            phonemes.chars().count()
        );
        return Vec::new();
    }
    let mid = words.len() / 2;
    let left = words[..mid].join(" ");
    let right = words[mid..].join(" ");
    let mut out = halve_by_words(&left, phonemizer, max);
    out.extend(halve_by_words(&right, phonemizer, max));
    out
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

#[cfg(test)]
mod fallback_tests {
    use super::phonemize_with_fallback;
    use crate::phonemizer::Phonemizer;
    use anyhow::Result;

    /// Test phonemizer that returns the input text unchanged, so char
    /// count IS phoneme count. Lets us exercise the splitter logic
    /// without a real phonemization model.
    struct EchoPhonemizer;
    impl Phonemizer for EchoPhonemizer {
        fn phonemize(&self, text: &str) -> Result<String> {
            Ok(text.to_string())
        }
    }

    #[test]
    fn under_cap_passes_through_as_single_piece() {
        let pieces = phonemize_with_fallback("hello world", &EchoPhonemizer, 510, 1);
        assert_eq!(pieces, vec!["hello world".to_string()]);
    }

    #[test]
    fn over_cap_resplits_on_commas() {
        // 300 chars + ", " + 300 chars = 602 chars > 510. Comma split
        // yields two pieces of 300 chars, both under the cap.
        let left = "a".repeat(300);
        let right = "b".repeat(300);
        let text = format!("{left}, {right}");
        let pieces = phonemize_with_fallback(&text, &EchoPhonemizer, 510, 1);
        assert_eq!(pieces.len(), 2);
        for p in &pieces {
            assert!(p.chars().count() <= 510, "piece too long: {} chars", p.len());
        }
    }

    #[test]
    fn comma_pieces_that_still_overflow_fall_back_to_word_halving() {
        // Build a long sub-piece (700 chars, comma-separated single
        // piece). Comma split returns one piece > cap; word-halving
        // takes over and breaks it into halves until each fits.
        let big = "word ".repeat(200); // 1000 chars, 200 words
        let text = format!("short prefix, {big}");
        let pieces = phonemize_with_fallback(&text, &EchoPhonemizer, 510, 1);
        assert!(pieces.len() >= 2);
        for p in &pieces {
            assert!(p.chars().count() <= 510, "piece too long: {}", p.chars().count());
        }
    }

    #[test]
    fn no_commas_halves_directly() {
        let text = "word ".repeat(200); // 1000 chars
        let pieces = phonemize_with_fallback(&text, &EchoPhonemizer, 510, 1);
        assert!(pieces.len() >= 2);
        for p in &pieces {
            assert!(p.chars().count() <= 510);
        }
    }

    #[test]
    fn single_huge_unsplittable_word_is_dropped() {
        let huge = "a".repeat(600);
        let pieces = phonemize_with_fallback(&huge, &EchoPhonemizer, 510, 1);
        assert!(pieces.is_empty());
    }

    #[test]
    fn empty_input_returns_empty() {
        let pieces = phonemize_with_fallback("", &EchoPhonemizer, 510, 1);
        assert!(pieces.is_empty());
    }

    #[test]
    fn index_style_long_list_splits_into_many_small_pieces() {
        // Models the index entry that broke chapter 146: a long
        // comma-separated list of page numbers (~970 phonemes).
        let entry = (1..=200)
            .map(|n| format!("{n}"))
            .collect::<Vec<_>>()
            .join(", ");
        let pieces = phonemize_with_fallback(&entry, &EchoPhonemizer, 510, 1);
        assert!(!pieces.is_empty());
        for p in &pieces {
            assert!(p.chars().count() <= 510);
        }
    }
}
