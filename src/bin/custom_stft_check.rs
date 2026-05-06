//! Validate CustomSTFT against `tools/reference_custom_stft.py`.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use kokoro_tts::model::stft::CustomStft;
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct Args {
    input: PathBuf,
    reference: PathBuf,
    n_fft: usize,
    hop: usize,
    atol: f32,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let mut parsed = Self {
            input: PathBuf::from("tmp/reference_custom_stft_input.bin"),
            reference: PathBuf::from("tmp/reference_custom_stft.bin"),
            n_fft: 20,
            hop: 5,
            atol: 1e-4,
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--input" => parsed.input = PathBuf::from(args.next().context("--input requires a path")?),
                "--ref" => parsed.reference = PathBuf::from(args.next().context("--ref requires a path")?),
                "--n-fft" => parsed.n_fft = args.next().context("--n-fft requires a value")?.parse()?,
                "--hop" => parsed.hop = args.next().context("--hop requires a value")?.parse()?,
                "--atol" => parsed.atol = args.next().context("--atol requires a value")?.parse()?,
                "--help" | "-h" => {
                    println!("usage: cargo run --bin custom_stft_check -- --input IN_BIN --ref OUT_BIN [--n-fft 20] [--hop 5] [--atol 1e-4]");
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
    let (input_shape, input_data) = read_f32_bin(&args.input)?;
    let (ref_shape, ref_data) = read_f32_bin(&args.reference)?;
    if input_shape.len() != 2 {
        bail!("expected rank-2 waveform input, got {input_shape:?}");
    }
    let waveform = Tensor::from_vec(input_data, (input_shape[0], input_shape[1]), &device)?;
    let stft = CustomStft::new(args.n_fft, args.hop, &device)?;
    let out = stft.forward(&waveform)?;
    compare(&out, &ref_shape, &ref_data, args.atol)?;
    println!("OK");
    Ok(())
}
