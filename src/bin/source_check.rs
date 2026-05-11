//! Validate SineGen + SourceModuleHnNSF against `tools/reference_source.py`.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use kokoro_tts::model::source::SourceModuleHnNsf;
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct Args {
    model_dir: PathBuf,
    f0: PathBuf,
    rand_ini: PathBuf,
    noise: PathBuf,
    reference: PathBuf,
    uv_reference: PathBuf,
    atol: f32,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let mut parsed = Self {
            model_dir: PathBuf::from("models"),
            f0: PathBuf::from("tmp/reference_source_f0.bin"),
            rand_ini: PathBuf::from("tmp/reference_source_rand_ini.bin"),
            noise: PathBuf::from("tmp/reference_source_noise.bin"),
            reference: PathBuf::from("tmp/reference_source.bin"),
            uv_reference: PathBuf::from("tmp/reference_source_uv.bin"),
            atol: 1e-3,
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model-dir" => {
                    parsed.model_dir =
                        PathBuf::from(args.next().context("--model-dir requires a path")?)
                }
                "--f0" => parsed.f0 = PathBuf::from(args.next().context("--f0 requires a path")?),
                "--rand" => {
                    parsed.rand_ini = PathBuf::from(args.next().context("--rand requires a path")?)
                }
                "--noise" => {
                    parsed.noise = PathBuf::from(args.next().context("--noise requires a path")?)
                }
                "--ref" => {
                    parsed.reference = PathBuf::from(args.next().context("--ref requires a path")?)
                }
                "--uv-ref" => {
                    parsed.uv_reference =
                        PathBuf::from(args.next().context("--uv-ref requires a path")?)
                }
                "--atol" => {
                    parsed.atol = args.next().context("--atol requires a value")?.parse()?
                }
                "--help" | "-h" => {
                    println!("usage: cargo run --bin source_check -- --model-dir models --f0 F0 --rand RAND --noise NOISE --ref REF --uv-ref UV_REF [--atol 1e-3]");
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

fn tensor_from_bin(path: &Path, device: &Device) -> Result<Tensor> {
    let (shape, data) = read_f32_bin(path)?;
    Ok(match shape.as_slice() {
        [a, b] => Tensor::from_vec(data, (*a, *b), device)?,
        [a, b, c] => Tensor::from_vec(data, (*a, *b, *c), device)?,
        _ => bail!("unsupported tensor rank in {}: {:?}", path.display(), shape),
    })
}

fn compare(
    name: &str,
    got: &Tensor,
    ref_shape: &[usize],
    ref_data: &[f32],
    atol: f32,
) -> Result<()> {
    if got.dims() != ref_shape {
        bail!(
            "{name} shape mismatch: rust {:?} vs ref {:?}",
            got.dims(),
            ref_shape
        );
    }
    let got_data = got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut argmax = 0usize;
    for (i, (&actual, &expected)) in got_data.iter().zip(ref_data).enumerate() {
        let delta = (actual - expected).abs();
        sum_abs += f64::from(delta);
        if delta > max_abs {
            max_abs = delta;
            argmax = i;
        }
    }
    let mean_abs = sum_abs / got_data.len() as f64;
    println!(
        "{name}: shape={:?} max_abs={:.3e} mean_abs={:.3e} argmax={} rust={:.8e} ref={:.8e}",
        got.dims(),
        max_abs,
        mean_abs,
        argmax,
        got_data[argmax],
        ref_data[argmax]
    );
    if max_abs > atol {
        bail!(
            "{name} FAIL: max_abs {:.3e} exceeds tolerance {:.3e}",
            max_abs,
            atol
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    let device = Device::Cpu;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[args.model_dir.join("model.safetensors")],
            DType::F32,
            &device,
        )?
    };
    let source = SourceModuleHnNsf::load(vb.pp("decoder.generator.m_source"))?;
    let f0 = tensor_from_bin(&args.f0, &device)?;
    let rand_ini = tensor_from_bin(&args.rand_ini, &device)?;
    let noise = tensor_from_bin(&args.noise, &device)?;
    let (ref_shape, ref_data) = read_f32_bin(&args.reference)?;
    let (uv_shape, uv_data) = read_f32_bin(&args.uv_reference)?;

    let (out, _, uv) = source.forward_with_controls(&f0, Some(&rand_ini), Some(&noise))?;
    compare("source", &out, &ref_shape, &ref_data, args.atol)?;
    compare("uv", &uv, &uv_shape, &uv_data, 0.0)?;
    println!("OK");
    Ok(())
}
