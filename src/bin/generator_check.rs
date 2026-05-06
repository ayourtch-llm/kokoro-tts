//! Validate iSTFTNet Generator against `tools/reference_generator.py`.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use kokoro_tts::model::generator::Generator;
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct Args {
    model_dir: PathBuf,
    x: PathBuf,
    s: PathBuf,
    f0: PathBuf,
    rand_ini: PathBuf,
    noise: PathBuf,
    reference: PathBuf,
    atol: f32,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let mut parsed = Self {
            model_dir: PathBuf::from("models"),
            x: PathBuf::from("tmp/reference_generator_x.bin"),
            s: PathBuf::from("tmp/reference_generator_s.bin"),
            f0: PathBuf::from("tmp/reference_generator_f0.bin"),
            rand_ini: PathBuf::from("tmp/reference_generator_rand_ini.bin"),
            noise: PathBuf::from("tmp/reference_generator_noise.bin"),
            reference: PathBuf::from("tmp/reference_generator.bin"),
            atol: 5e-3,
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model-dir" => parsed.model_dir = PathBuf::from(args.next().context("--model-dir")?),
                "--x" => parsed.x = PathBuf::from(args.next().context("--x")?),
                "--s" => parsed.s = PathBuf::from(args.next().context("--s")?),
                "--f0" => parsed.f0 = PathBuf::from(args.next().context("--f0")?),
                "--rand-ini" => parsed.rand_ini = PathBuf::from(args.next().context("--rand-ini")?),
                "--noise" => parsed.noise = PathBuf::from(args.next().context("--noise")?),
                "--ref" => parsed.reference = PathBuf::from(args.next().context("--ref")?),
                "--atol" => parsed.atol = args.next().context("--atol")?.parse()?,
                "--help" | "-h" => {
                    println!("usage: cargo run --bin generator_check -- [--model-dir DIR] [--x ...] [--s ...] [--f0 ...] [--rand-ini ...] [--noise ...] [--ref ...] [--atol T]");
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
        bail!("{} too short", path.display());
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

fn tensor_from(path: &Path, device: &Device) -> Result<(Vec<usize>, Tensor)> {
    let (shape, data) = read_f32_bin(path)?;
    let t = match shape.len() {
        2 => Tensor::from_vec(data, (shape[0], shape[1]), device)?,
        3 => Tensor::from_vec(data, (shape[0], shape[1], shape[2]), device)?,
        n => bail!("unsupported rank {n} in {}", path.display()),
    };
    Ok((shape, t))
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    let device = Device::Cpu;

    let (_, x) = tensor_from(&args.x, &device)?;
    let (_, s) = tensor_from(&args.s, &device)?;
    let (_, f0) = tensor_from(&args.f0, &device)?;
    let (_, rand_ini) = tensor_from(&args.rand_ini, &device)?;
    let (_, noise) = tensor_from(&args.noise, &device)?;
    let (ref_shape, ref_data) = read_f32_bin(&args.reference)?;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[args.model_dir.join("model.safetensors")],
            DType::F32,
            &device,
        )?
    };
    let gen = Generator::load(
        128,
        512,
        vec![10, 6],
        vec![20, 12],
        vec![3, 7, 11],
        vec![[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        20,
        5,
        vb.pp("decoder.generator"),
    )?;
    let out = gen.forward_with_controls(&x, &s, &f0, Some(&rand_ini), Some(&noise))?;

    if out.dims() != ref_shape {
        bail!("shape mismatch: rust {:?} vs ref {:?}", out.dims(), ref_shape);
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
    let max_ref = ref_data.iter().map(|v| v.abs()).fold(0f32, f32::max);
    println!(
        "shape={:?} max_abs={:.3e} mean_abs={:.3e} max_ref={:.3e} argmax={} rust={:.6e} ref={:.6e}",
        out.dims(),
        max_abs,
        mean_abs,
        max_ref,
        argmax,
        out_data[argmax],
        ref_data[argmax]
    );
    if max_abs > args.atol * max_ref.max(1.0) {
        bail!("FAIL: max_abs {:.3e} exceeds {:.3e} (atol {:.3e} * max_ref {:.3e})",
            max_abs, args.atol * max_ref.max(1.0), args.atol, max_ref);
    }
    println!("OK");
    Ok(())
}
