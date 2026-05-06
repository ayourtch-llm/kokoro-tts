use crate::model::Kokoro;
use crate::phonemizer::Phonemizer;
use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use std::fs;
use std::net::{SocketAddr, UdpSocket};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const SILENCE_PADDING_SAMPLES: usize = 24_000 * 80 / 1_000;
pub const MAX_SENTENCE_PHONEMES: usize = 510;
pub const REFERENCE_UDP_SAMPLE_RATE: u32 = 16_000;
pub const REFERENCE_UDP_CHUNK_SAMPLES: usize = 320;

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

pub fn send_reference_audio(
    socket: &UdpSocket,
    target: SocketAddr,
    samples_24k: &[f32],
    silence_padding_samples: usize,
) -> Result<(usize, usize)> {
    if samples_24k.is_empty() {
        return Ok((0, 0));
    }
    let mut audio = samples_24k.to_vec();
    audio.extend(std::iter::repeat(0.0).take(silence_padding_samples));
    let resampled = crate::audio::resample_linear(&audio, 24_000, REFERENCE_UDP_SAMPLE_RATE);
    let mut packet = vec![0u8; REFERENCE_UDP_CHUNK_SAMPLES * 4];
    let mut packets = 0usize;
    for chunk in resampled.chunks(REFERENCE_UDP_CHUNK_SAMPLES) {
        packet.fill(0);
        for (idx, sample) in chunk.iter().enumerate() {
            let le = sample.to_le_bytes();
            let byte_idx = idx * 4;
            packet[byte_idx..byte_idx + 4].copy_from_slice(&le);
        }
        socket
            .send_to(&packet, target)
            .with_context(|| format!("sending reference packet to {target}"))?;
        packets += 1;
    }
    Ok((packets, packets * packet.len()))
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
