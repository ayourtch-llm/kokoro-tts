//! Validate Generator Snake1D against `tools/reference_snake.py`.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use kokoro_tts::model::generator::Snake1d;
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct Args {
    model_dir: PathBuf,
    input: PathBuf,
    reference: PathBuf,
    alpha_key: String,
    channels: usize,
    atol: f32,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let mut parsed = Self {
            model_dir: PathBuf::from("models"),
            input: PathBuf::from("tmp/reference_snake_input.bin"),
            reference: PathBuf::from("tmp/reference_snake.bin"),
            alpha_key: "decoder.generator.resblocks.0.alpha1.0".to_string(),
            channels: 256,
            atol: 1e-6,
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model-dir" => {
                    parsed.model_dir =
                        PathBuf::from(args.next().context("--model-dir requires a path")?)
                }
                "--input" => {
                    parsed.input = PathBuf::from(args.next().context("--input requires a path")?)
                }
                "--ref" => {
                    parsed.reference = PathBuf::from(args.next().context("--ref requires a path")?)
                }
                "--alpha-key" => {
                    parsed.alpha_key = args.next().context("--alpha-key requires a value")?
                }
                "--channels" => {
                    parsed.channels = args
                        .next()
                        .context("--channels requires a value")?
                        .parse()?
                }
                "--atol" => {
                    parsed.atol = args.next().context("--atol requires a value")?.parse()?
                }
                "--help" | "-h" => {
                    println!("usage: cargo run --bin snake_check -- --model-dir models --input IN --ref REF --alpha-key KEY [--channels 256] [--atol 1e-6]");
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
        shape.push(u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize);
        offset += 4;
    }
    let nelem: usize = shape.iter().product();
    if bytes.len() != offset + nelem * 4 {
        bail!("{} size mismatch for shape {:?}", path.display(), shape);
    }
    let mut data = Vec::with_capacity(nelem);
    for chunk in bytes[offset..].chunks_exact(4) {
        data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok((shape, data))
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    let device = Device::Cpu;
    let (input_shape, input_data) = read_f32_bin(&args.input)?;
    let (ref_shape, ref_data) = read_f32_bin(&args.reference)?;
    if input_shape.len() != 3 {
        bail!("expected rank-3 input, got {input_shape:?}");
    }
    let input = Tensor::from_vec(
        input_data,
        (input_shape[0], input_shape[1], input_shape[2]),
        &device,
    )?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[args.model_dir.join("model.safetensors")],
            DType::F32,
            &device,
        )?
    };
    let snake = Snake1d::load_named(args.channels, vb, &args.alpha_key)?;
    let out = snake.forward(&input)?;
    if out.dims() != ref_shape {
        bail!(
            "shape mismatch: rust {:?} vs ref {:?}",
            out.dims(),
            ref_shape
        );
    }
    let out_data = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut argmax = 0usize;
    for (i, (&actual, &expected)) in out_data.iter().zip(&ref_data).enumerate() {
        let delta = (actual - expected).abs();
        sum_abs += f64::from(delta);
        if delta > max_abs {
            max_abs = delta;
            argmax = i;
        }
    }
    let mean_abs = sum_abs / out_data.len() as f64;
    println!(
        "shape={:?} max_abs={:.3e} mean_abs={:.3e} argmax={} rust={:.8e} ref={:.8e}",
        out.dims(),
        max_abs,
        mean_abs,
        argmax,
        out_data[argmax],
        ref_data[argmax]
    );
    if max_abs > args.atol {
        bail!(
            "FAIL: max_abs {:.3e} exceeds tolerance {:.3e}",
            max_abs,
            args.atol
        );
    }
    println!("OK");
    Ok(())
}
