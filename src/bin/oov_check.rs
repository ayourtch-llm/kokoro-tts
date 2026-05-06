//! Validate stage 5 OOV letter-to-sound rules against `tools/reference_oov.py`.

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
            reference: PathBuf::from("tmp/reference_oov.jsonl"),
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--ref" => {
                    parsed.reference = PathBuf::from(args.next().context("--ref requires a path")?)
                }
                "--help" | "-h" => {
                    println!("usage: cargo run --bin oov_check -- --ref tmp/reference_oov.jsonl");
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
    let mut weighted_similarity = 0.0f64;
    let mut first_mismatch = None;

    for line in reference.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let record: Record = serde_json::from_str(line)
            .with_context(|| format!("parsing reference line: {line}"))?;
        let got = phonemizer.phonemize(&record.case)?;
        let got_norm = normalize_ipa(&got);
        let expected_norm = normalize_ipa(&record.ipa);
        let score = similarity(&got_norm, &expected_norm);
        weighted_similarity += score;
        if score < 1.0 && first_mismatch.is_none() {
            first_mismatch = Some((record.case, got, record.ipa, score));
        }
        total += 1;
    }

    let avg = if total == 0 {
        0.0
    } else {
        weighted_similarity / total as f64
    };
    println!("total={total} avg_similarity={avg:.3}");
    if let Some((case, got, expected, score)) = first_mismatch {
        println!("first mismatch: {case:?} score={score:.3}");
        println!("  rust={got:?}");
        println!("  ref ={expected:?}");
    }
    if avg < 0.70 {
        bail!("agreement below threshold: {:.3}", avg);
    }
    Ok(())
}

fn normalize_ipa(text: &str) -> String {
    text.chars()
        .filter(|ch| !ch.is_whitespace() && *ch != '_' && *ch != 'ː')
        .collect()
}

fn similarity(a: &str, b: &str) -> f64 {
    let dist = levenshtein(a, b) as f64;
    let denom = a.len().max(b.len()) as f64;
    if denom == 0.0 {
        1.0
    } else {
        (1.0 - dist / denom).max(0.0)
    }
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

#[derive(Debug, Deserialize)]
struct Record {
    case: String,
    ipa: String,
}
