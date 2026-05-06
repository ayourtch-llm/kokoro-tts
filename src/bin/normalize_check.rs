//! Validate stage-3 cardinal-number normalization against `tools/reference_normalize.py`.

use anyhow::{bail, Context, Result};
use kokoro_tts::phonemizer::{
    normalize_abbreviations, normalize_acronyms, normalize_cardinals, normalize_dates,
    normalize_money_time, normalize_units,
};
use std::path::PathBuf;

#[derive(Debug)]
struct Args {
    reference: PathBuf,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let mut parsed = Self {
            reference: PathBuf::from("tmp/reference_normalize.tsv"),
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--ref" => {
                    parsed.reference = PathBuf::from(args.next().context("--ref requires a path")?)
                }
                "--help" | "-h" => {
                    println!(
                        "usage: cargo run --bin normalize_check -- --ref tmp/reference_normalize.tsv"
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
    let mut n = 0usize;
    for line in reference.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let Some((case, expected)) = line.split_once('\t') else {
            bail!("malformed reference line: {line}");
        };
        let got = normalize_cardinals(&normalize_acronyms(&normalize_units(
            &normalize_money_time(&normalize_dates(&normalize_abbreviations(case))),
        )));
        if got != expected {
            bail!("mismatch for {case:?}: rust={:?} ref={:?}", got, expected);
        }
        n += 1;
        println!("{case}: OK");
    }
    println!("OK ({n} cases)");
    Ok(())
}
