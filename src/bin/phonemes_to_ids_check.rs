//! Validate stage 2 phoneme-to-vocab mapping against `tools/reference_phonemes_to_ids.py`.

use anyhow::{bail, Context, Result};
use kokoro_tts::model::Config;
use kokoro_tts::phonemizer::MILESTONE_TEST_PHONEMES;
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct Args {
    config: PathBuf,
    phonemes: String,
    reference: PathBuf,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let mut parsed = Self {
            config: PathBuf::from("models/config.json"),
            phonemes: MILESTONE_TEST_PHONEMES.to_string(),
            reference: PathBuf::from("tmp/reference_phoneme_ids.bin"),
        };

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--config" => {
                    parsed.config = PathBuf::from(args.next().context("--config requires a path")?)
                }
                "--phonemes" => {
                    parsed.phonemes = args.next().context("--phonemes requires a value")?
                }
                "--ref" => {
                    parsed.reference = PathBuf::from(args.next().context("--ref requires a path")?)
                }
                "--help" | "-h" => {
                    println!(
                        "usage: cargo run --bin phonemes_to_ids_check -- --config PATH --phonemes IPA --ref PATH"
                    );
                    std::process::exit(0);
                }
                other => bail!("unknown argument {other}"),
            }
        }
        Ok(parsed)
    }
}

fn read_i64_bin(path: &Path) -> Result<Vec<i64>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.len() < 8 {
        bail!("{} is too short", path.display());
    }
    let ndim = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    if ndim != 1 {
        bail!("expected rank-1 ids tensor, got rank {ndim}");
    }
    let len = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let expected_len = 8 + len * 8;
    if bytes.len() != expected_len {
        bail!(
            "{} size mismatch: shape [{len}] expects {} bytes, got {}",
            path.display(),
            expected_len,
            bytes.len()
        );
    }
    let mut out = Vec::with_capacity(len);
    for chunk in bytes[8..].chunks_exact(8) {
        out.push(i64::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(out)
}

fn phonemes_to_ids(config: &Config, phonemes: &str) -> Vec<i64> {
    phonemes
        .chars()
        .filter_map(|phoneme| config.vocab.get(&phoneme.to_string()).map(|id| *id as i64))
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    let config = Config::load(&args.config).map_err(|e| anyhow::anyhow!("{e:#}"))?;
    let reference = read_i64_bin(&args.reference)?;
    let got = phonemes_to_ids(&config, &args.phonemes);

    println!("ids={got:?}");
    if got != reference {
        bail!("id mismatch: rust {got:?} vs ref {reference:?}");
    }
    println!("exact match len={}", got.len());
    println!("OK");
    Ok(())
}
