//! Validate stage-1 CMUdict phonemizer against `tools/reference_phonemize_lexicon.py`.

use anyhow::{bail, Context, Result};
use kokoro_tts::phonemizer::{Phonemizer, TwoTierPhonemizer};
use std::path::PathBuf;

#[derive(Debug)]
struct Args {
    reference: PathBuf,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let mut parsed = Self {
            reference: PathBuf::from("tmp/reference_lexicon.tsv"),
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--ref" => {
                    parsed.reference = PathBuf::from(args.next().context("--ref requires a path")?)
                }
                "--help" | "-h" => {
                    println!(
                        "usage: cargo run --bin lexicon_check -- --ref tmp/reference_lexicon.tsv"
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
        let Some((case, expected)) = line.split_once('\t') else {
            bail!("malformed reference line: {line}");
        };
        let got = phonemizer.phonemize(case)?;
        if got != expected {
            bail!("mismatch for {case:?}: rust={:?} ref={:?}", got, expected);
        }
        n += 1;
        println!("{case}: OK");
    }
    println!("OK ({n} cases)");
    Ok(())
}
