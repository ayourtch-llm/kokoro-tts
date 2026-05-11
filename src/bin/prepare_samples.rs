//! Walk a directory tree of plain-text files, run each file through
//! kokoro's `split_sentences`, and write one sentence per line to an
//! output file. Input for the TTS-ASR accuracy harness; intended to
//! exercise the sentence splitter on a wide corpus (e.g. a Calibre
//! library export).
//!
//! Usage:
//!   cargo run --release --bin prepare-samples -- \
//!     --dir "~/kindle-backup/Calibre Library" \
//!     --out tmp/samples.txt

use anyhow::{Context, Result};
use kokoro_tts::phonemizer::sentence::split_sentences;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

struct Args {
    dir: PathBuf,
    out: PathBuf,
    max_files: Option<usize>,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut dir = None;
        let mut out = PathBuf::from("tmp/samples.txt");
        let mut max_files = None;
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--dir" => dir = Some(PathBuf::from(it.next().context("--dir needs value")?)),
                "--out" => out = PathBuf::from(it.next().context("--out needs value")?),
                "--max-files" => {
                    max_files = Some(it.next().context("--max-files needs value")?.parse()?)
                }
                "-h" | "--help" => {
                    println!(
                        "usage: prepare-samples --dir DIR --out FILE [--max-files N]\n\
                         walks DIR for *.txt, splits each into sentences, writes one per line"
                    );
                    std::process::exit(0);
                }
                other => anyhow::bail!("unknown arg {other}"),
            }
        }
        Ok(Self {
            dir: dir.context("--dir is required")?,
            out,
            max_files,
        })
    }
}

fn collect_txt(root: &Path, out: &mut Vec<PathBuf>, limit: Option<usize>) -> Result<()> {
    if let Some(n) = limit {
        if out.len() >= n {
            return Ok(());
        }
    }
    let meta = fs::metadata(root)
        .with_context(|| format!("stat {}", root.display()))?;
    if meta.is_file() {
        if root.extension().and_then(|s| s.to_str()) == Some("txt") {
            out.push(root.to_path_buf());
        }
        return Ok(());
    }
    let mut entries: Vec<_> = fs::read_dir(root)
        .with_context(|| format!("read_dir {}", root.display()))?
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        collect_txt(&entry.path(), out, limit)?;
        if let Some(n) = limit {
            if out.len() >= n {
                break;
            }
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    let mut files = Vec::new();
    collect_txt(&args.dir, &mut files, args.max_files)?;
    if let Some(n) = args.max_files {
        files.truncate(n);
    }
    println!("found {} .txt files under {}", files.len(), args.dir.display());

    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let mut writer = std::io::BufWriter::new(
        fs::File::create(&args.out)
            .with_context(|| format!("creating {}", args.out.display()))?,
    );

    let mut total_sentences = 0usize;
    let mut total_bytes = 0u64;
    let mut empty_files = 0usize;
    let mut errored_files = 0usize;
    for (i, path) in files.iter().enumerate() {
        let text = match fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("warn: read {}: {e}", path.display());
                errored_files += 1;
                continue;
            }
        };
        if text.trim().is_empty() {
            empty_files += 1;
            continue;
        }
        total_bytes += text.len() as u64;
        let sentences = split_sentences(&text);
        for s in &sentences {
            // The splitter already trims and skips empty fragments, but be
            // defensive: any newline inside a sentence would corrupt the
            // one-per-line file format.
            let line: String = s
                .chars()
                .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
                .collect();
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            writeln!(writer, "{line}")?;
            total_sentences += 1;
        }
        if (i + 1) % 50 == 0 || i + 1 == files.len() {
            eprintln!(
                "  [{:>4}/{}] {} sentences so far",
                i + 1,
                files.len(),
                total_sentences
            );
        }
    }
    writer.flush()?;
    println!(
        "wrote {} sentences ({:.1} MB of input text, {} empty files, {} read errors) to {}",
        total_sentences,
        total_bytes as f64 / 1_048_576.0,
        empty_files,
        errored_files,
        args.out.display(),
    );
    Ok(())
}
