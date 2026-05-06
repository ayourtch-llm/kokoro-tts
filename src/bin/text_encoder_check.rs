//! Validate stage 5 TextEncoder against `tools/reference_text_encoder.py`.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use kokoro_tts::model::text_encoder::TextEncoder;
use kokoro_tts::model::Config;
use kokoro_tts::phonemizer::MILESTONE_TEST_PHONEMES;
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct Args {
    model: PathBuf,
    config: PathBuf,
    phonemes: String,
    reference: PathBuf,
    atol: f32,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let mut parsed = Self {
            model: PathBuf::from("models/model.safetensors"),
            config: PathBuf::from("models/config.json"),
            phonemes: MILESTONE_TEST_PHONEMES.to_string(),
            reference: PathBuf::from("tmp/reference_text_encoder.bin"),
            atol: 1e-4,
        };

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model" => {
                    parsed.model = PathBuf::from(args.next().context("--model requires a path")?)
                }
                "--config" => {
                    parsed.config = PathBuf::from(args.next().context("--config requires a path")?)
                }
                "--phonemes" => {
                    parsed.phonemes = args.next().context("--phonemes requires a value")?
                }
                "--ref" => {
                    parsed.reference =
                        PathBuf::from(args.next().context("--ref requires a path")?)
                }
                "--atol" => {
                    parsed.atol = args
                        .next()
                        .context("--atol requires a value")?
                        .parse()
                        .context("parsing --atol")?;
                }
                "--help" | "-h" => {
                    println!(
                        "usage: cargo run --bin text_encoder_check -- --model PATH --config PATH --ref PATH [--phonemes IPA] [--atol F32]"
                    );
                    std::process::exit(0);
                }
                other => bail!("unknown argument {other}"),
            }
        }
        Ok(parsed)
    }
}

fn read_f32_bin(path: &Path) -> Result<(Vec<usize>, Vec<f32>)> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.len() < 4 {
        bail!("{} is too short", path.display());
    }
    let ndim = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let mut shape = Vec::with_capacity(ndim);
    let mut offset = 4;
    for _ in 0..ndim {
        if offset + 4 > bytes.len() {
            bail!("{} has a truncated shape header", path.display());
        }
        shape.push(u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize);
        offset += 4;
    }
    let nelem: usize = shape.iter().product();
    if bytes.len() != offset + nelem * 4 {
        bail!(
            "{} size mismatch: shape {:?} expects {} f32 values",
            path.display(),
            shape,
            nelem
        );
    }
    let mut data = Vec::with_capacity(nelem);
    for chunk in bytes[offset..].chunks_exact(4) {
        data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok((shape, data))
}

fn phonemes_to_ids(config: &Config, phonemes: &str) -> Vec<i64> {
    phonemes
        .chars()
        .filter_map(|phoneme| config.vocab.get(&phoneme.to_string()).map(|id| *id as i64))
        .collect()
}

fn compare(rust: &Tensor, ref_shape: &[usize], ref_data: &[f32], atol: f32) -> Result<()> {
    if rust.dims() != ref_shape {
        bail!("shape mismatch: rust {:?} vs ref {:?}", rust.dims(), ref_shape);
    }
    let rust_data = rust.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut argmax = 0usize;
    for (i, (&got, &expected)) in rust_data.iter().zip(ref_data).enumerate() {
        let delta = (got - expected).abs();
        sum_abs += f64::from(delta);
        if delta > max_abs {
            max_abs = delta;
            argmax = i;
        }
    }
    let mean_abs = sum_abs / rust_data.len() as f64;
    println!(
        "shape={:?} max_abs={:.3e} mean_abs={:.3e} argmax={} rust={:.8e} ref={:.8e}",
        rust.dims(),
        max_abs,
        mean_abs,
        argmax,
        rust_data[argmax],
        ref_data[argmax]
    );
    if max_abs > atol {
        bail!("FAIL: max_abs {:.3e} exceeds tolerance {:.3e}", max_abs, atol);
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    let device = Device::Cpu;
    let config = Config::load(&args.config).map_err(|e| anyhow::anyhow!("{e:#}"))?;
    let (ref_shape, ref_data) = read_f32_bin(&args.reference)?;

    let mut ids = Vec::with_capacity(args.phonemes.chars().count() + 2);
    ids.push(0);
    ids.extend(phonemes_to_ids(&config, &args.phonemes));
    ids.push(0);
    let input_ids = Tensor::from_vec(ids.clone(), (1, ids.len()), &device)?;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[args.model.clone()], DType::F32, &device)
            .context("loading model safetensors")?
    };
    let text_encoder = TextEncoder::load(
        config.hidden_dim,
        config.text_encoder_kernel_size,
        config.n_layer,
        config.n_token,
        vb.pp("text_encoder"),
    )
    .map_err(|e| anyhow::anyhow!("{e:#}"))?;

    let out = text_encoder
        .forward(&input_ids)
        .map_err(|e| anyhow::anyhow!("{e:#}"))?;
    compare(&out, &ref_shape, &ref_data, args.atol)?;
    println!("OK");
    Ok(())
}
