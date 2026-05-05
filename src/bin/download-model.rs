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

    // Download default voice (af_heart)
    let voice_path =
        download_file(&repo, "voices/af_heart.pt", &output_path.join("voices")).await?;
    println!("Downloaded af_heart voice -> {}", voice_path.display());

    println!("\n=== Download complete ===");
    println!("\nNext step: convert .pth to safetensors for Candle.");
    println!("Run:");
    println!("  pip install torch safetensors huggingface_hub");
    println!(
        "  python scripts/convert_weights.py --repo {REPO_ID} --output {}",
        output_path.display()
    );
    println!("\nOr if you don't have Python, convert manually with a Python environment.");

    Ok(())
}

async fn download_file(repo: &ApiRepo, filename: &str, local_dir: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(local_dir)?;

    let dest = local_dir.join(filename);
    if dest.exists() {
        println!("  (already exists, skipping) {filename}");
        return Ok(dest);
    }

    println!("  Downloading {filename}...");
    let path = repo
        .get(filename)
        .context(format!("Failed to download {filename}"))?;
    Ok(path)
}
