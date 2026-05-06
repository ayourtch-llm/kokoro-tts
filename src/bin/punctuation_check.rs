//! Validate stage-2 sentence-aware phonemizer against `tools/reference_punctuation.py`.

use anyhow::{bail, Context, Result};
use kokoro_tts::phonemizer::{Phonemizer, TwoTierPhonemizer};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug)]
struct Args {
    reference: PathBuf,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let mut parsed = Self {
            reference: PathBuf::from("tmp/reference_punctuation.jsonl"),
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--ref" => {
                    parsed.reference = PathBuf::from(args.next().context("--ref requires a path")?)
                }
                "--help" | "-h" => {
                    println!(
                        "usage: cargo run --bin punctuation_check -- --ref tmp/reference_punctuation.jsonl"
                    );
                    std::process::exit(0);
                }
                other => bail!("unknown argument {other}"),
            }
        }
        Ok(parsed)
    }
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    let reference = std::fs::read_to_string(&args.reference)
        .with_context(|| format!("reading {}", args.reference.display()))?;
    let phonemizer = TwoTierPhonemizer;
    let mut n = 0usize;
    for line in reference.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let record: Record = serde_json::from_str(line)
            .with_context(|| format!("parsing reference line: {line}"))?;
        let Record {
            case,
            ipa: expected,
        } = record;
        let got = phonemizer.phonemize(&case)?;
        if got != expected {
            bail!("mismatch for {:?}: rust={:?} ref={:?}", case, got, expected);
        }
        n += 1;
        println!("{}: OK", case);
    }
    println!("OK ({n} cases)");
    Ok(())
}

#[derive(Debug, Deserialize)]
struct Record {
    case: String,
    ipa: String,
}
