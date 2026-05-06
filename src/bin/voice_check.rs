//! Validate stage 1 voice tensor selection against `tools/reference_voice.py`.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct Args {
    voice: PathBuf,
    phoneme_count: usize,
    reference: PathBuf,
    atol: f32,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let mut parsed = Self {
            voice: PathBuf::from("models/voices/af_heart.safetensors"),
            phoneme_count: 11,
            reference: PathBuf::from("tmp/reference_voice.bin"),
            atol: 0.0,
        };

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--voice" => {
                    parsed.voice = PathBuf::from(args.next().context("--voice requires a path")?)
                }
                "--phoneme-count" => {
                    parsed.phoneme_count = args
                        .next()
                        .context("--phoneme-count requires a value")?
                        .parse()
                        .context("parsing --phoneme-count")?;
                }
                "--ref" => {
                    parsed.reference = PathBuf::from(args.next().context("--ref requires a path")?)
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
                        "usage: cargo run --bin voice_check -- --voice PATH --phoneme-count N --ref PATH [--atol F32]"
                    );
                    std::process::exit(0);
                }
                other => bail!("unknown argument {other}"),
            }
        }
        Ok(parsed)
    }
}

fn read_bin(path: &Path) -> Result<(Vec<usize>, Vec<f32>)> {
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

fn select_voice_style(path: &Path, phoneme_count: usize, device: &Device) -> Result<Tensor> {
    let tensors = candle_core::safetensors::load(path, device)
        .with_context(|| format!("loading {}", path.display()))?;
    let ref_s = tensors
        .get("ref_s")
        .with_context(|| format!("missing ref_s in {}", path.display()))?;
    let n = ref_s.dim(0)?;
    let idx = phoneme_count.min(n.saturating_sub(1));
    Ok(ref_s.narrow(0, idx, 1)?.squeeze(0)?)
}

fn compare(rust: &Tensor, ref_shape: &[usize], ref_data: &[f32], atol: f32) -> Result<()> {
    if rust.dims() != ref_shape {
        bail!(
            "shape mismatch: rust {:?} vs ref {:?}",
            rust.dims(),
            ref_shape
        );
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
        bail!(
            "FAIL: max_abs {:.3e} exceeds tolerance {:.3e}",
            max_abs,
            atol
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    let device = Device::Cpu;
    let (ref_shape, ref_data) = read_bin(&args.reference)?;
    let style = select_voice_style(&args.voice, args.phoneme_count, &device)?;
    compare(&style, &ref_shape, &ref_data, args.atol)?;
    println!("OK");
    Ok(())
}
