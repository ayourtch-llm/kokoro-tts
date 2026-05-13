//! Probe the chapter splitter against a real .txt book. Reports the
//! detected count and the first non-blank line of each chapter, so we
//! can eyeball whether the heuristic split where a human would.

use anyhow::{Context, Result};
use kokoro_tts::chapters::split_chapters;
use std::fs;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: probe_chapters <path-to-text-file> [<more paths>...]");
        std::process::exit(2);
    }
    for path in &args {
        let path = PathBuf::from(path);
        let text = fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let chapters = split_chapters(&text);
        println!(
            "\n=== {} ({} bytes, {} chapters) ===",
            path.display(),
            text.len(),
            chapters.len()
        );
        for (idx, ch) in chapters.iter().enumerate() {
            let preview = ch
                .lines()
                .filter(|l| !l.trim().is_empty())
                .take(3)
                .map(|l| l.trim())
                .collect::<Vec<_>>()
                .join(" / ");
            let preview: String = preview.chars().take(120).collect();
            println!("  {:>4}. [{} chars] {preview}", idx + 1, ch.len());
        }
    }
    Ok(())
}
