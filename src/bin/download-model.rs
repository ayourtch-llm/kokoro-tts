//! Download Kokoro model weights from HuggingFace Hub.
//!
//! Usage:
//!   cargo run --bin download-model [-- --path ./models]
//!
//! The model is downloaded to the specified path (default: ./models).
//! After download, run the Python conversion script to convert .pth to safetensors:
//!   pip install torch safetensors huggingface_hub
//!   python scripts/convert_weights.py --repo hexgrad/Kokoro-82M --output ./models

use anyhow::{Context, Result};
use hf_hub::{
    api::sync::{ApiBuilder, ApiRepo},
    Repo, RepoType,
};
use std::path::{Path, PathBuf};

const REPO_ID: &str = "hexgrad/Kokoro-82M";
const MODEL_FILE: &str = "kokoro-v1_0.pth";
const CONFIG_FILE: &str = "config.json";

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();
    let mut output_path = PathBuf::from("./models");

    for i in 0..args.len() {
        if args[i] == "--path" && i + 1 < args.len() {
            output_path = PathBuf::from(&args[i + 1]);
        }
    }

    println!("Downloading Kokoro model to: {}", output_path.display());

    let api = ApiBuilder::new().build()?;
    let repo = api.repo(Repo::new(REPO_ID.to_string(), RepoType::Model));

    // Download config.json
    let config_path = download_file(&repo, CONFIG_FILE, &output_path).await?;
    println!("Downloaded config.json -> {}", config_path.display());

    // Download model weights
    let model_path = download_file(&repo, MODEL_FILE, &output_path).await?;
    println!("Downloaded {} -> {}", MODEL_FILE, model_path.display());

    // Voices to fetch. Pulls the standard English-language set plus a
    // few non-English starters; users can extend with --voices.
    let voices: Vec<String> = {
        let mut v: Vec<String> = args
            .windows(2)
            .filter_map(|w| (w[0] == "--voices").then(|| w[1].clone()))
            .next()
            .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
            .unwrap_or_else(|| {
                vec![
                    // American female
                    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
                    "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
                    // American male
                    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
                    "am_michael", "am_onyx", "am_puck", "am_santa",
                    // British female
                    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
                    // British male
                    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
                ]
                .into_iter()
                .map(String::from)
                .collect()
            });
        v.sort();
        v.dedup();
        v
    };
    println!("Fetching {} voice files...", voices.len());
    let voices_dir = output_path.join("voices");
    let mut fetched = 0;
    for v in &voices {
        let rel = format!("voices/{v}.pt");
        match download_file(&repo, &rel, &voices_dir).await {
            Ok(p) => {
                fetched += 1;
                println!("  {v}.pt -> {}", p.display());
            }
            Err(e) => eprintln!("  WARN failed {v}.pt: {e:#}"),
        }
    }
    println!("Fetched {fetched}/{} voices.", voices.len());

    println!("\n=== Download complete ===");
    println!("\nNext steps:");
    println!("  1. Convert the main model .pth → safetensors (needs Python):");
    println!("       pip install torch safetensors huggingface_hub");
    println!(
        "       python scripts/convert_weights.py --input {} --output {}",
        output_path.display(),
        output_path.display()
    );
    println!("       (this now converts every voice in {}/voices/ in one go.)",
        output_path.display());

    Ok(())
}

async fn download_file(repo: &ApiRepo, filename: &str, local_dir: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(local_dir)?;

    // Strip any parent dir components from `filename` for the local destination
    // (e.g. "voices/af_heart.pt" → "af_heart.pt").
    let base_name = Path::new(filename)
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("no file name in {filename}"))?;
    let dest = local_dir.join(base_name);
    if dest.exists() {
        return Ok(dest);
    }

    let cache_path = repo
        .get(filename)
        .context(format!("Failed to download {filename}"))?;
    // Copy from HF cache to the local models dir so users have a stable
    // path to point --voice at and to feed the Python converter.
    std::fs::copy(&cache_path, &dest)
        .with_context(|| format!("copying {} → {}", cache_path.display(), dest.display()))?;
    Ok(dest)
}
