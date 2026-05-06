//! Validate stage 4 homograph disambiguation against `tools/reference_homograph.py`.

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
            reference: PathBuf::from("tmp/reference_homograph.jsonl"),
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--ref" => {
                    parsed.reference = PathBuf::from(args.next().context("--ref requires a path")?)
                }
                "--help" | "-h" => {
                    println!(
                        "usage: cargo run --bin homograph_check -- --ref tmp/reference_homograph.jsonl"
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
    let mut total = 0usize;
    let mut exact = 0usize;
    let mut first_mismatch = None;
    for line in reference.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let record: Record = serde_json::from_str(line)
            .with_context(|| format!("parsing reference line: {line}"))?;
        let got = phonemizer.phonemize(&record.case)?;
        total += 1;
        if got == record.ipa {
            exact += 1;
        } else if first_mismatch.is_none() {
            first_mismatch = Some((record.case, got, record.ipa));
        }
    }

    let accuracy = if total == 0 {
        0.0
    } else {
        exact as f64 / total as f64
    };
    println!("total={total} exact={exact} accuracy={accuracy:.3}");
    if let Some((case, got, expected)) = first_mismatch {
        println!("first mismatch: {case:?}");
        println!("  rust={got:?}");
        println!("  ref ={expected:?}");
    }
    if accuracy < 0.80 {
        bail!("agreement below threshold: {:.3}", accuracy);
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct Record {
    case: String,
    ipa: String,
}
