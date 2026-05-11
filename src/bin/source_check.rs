//! Validate SineGen + SourceModuleHnNSF against `tools/reference_source.py`.
//! Phase E reference capture:
//! `cargo run --bin source_check -- --dump-dir /tmp/source_ref`

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
    source_reference: Option<PathBuf>,
    noise_reference: Option<PathBuf>,
    uv_reference: PathBuf,
    atol: f32,
    mean_atol: Option<f64>,
    dump_dir: Option<PathBuf>,
    device: String,
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
            source_reference: None,
            noise_reference: None,
            uv_reference: PathBuf::from("tmp/reference_source_uv.bin"),
            atol: 1e-3,
            mean_atol: None,
            dump_dir: None,
            device: "cpu".to_string(),
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
                "--source-ref" => {
                    parsed.source_reference =
                        Some(PathBuf::from(args.next().context("--source-ref requires a path")?))
                }
                "--noise-ref" => {
                    parsed.noise_reference =
                        Some(PathBuf::from(args.next().context("--noise-ref requires a path")?))
                }
                "--uv-ref" => {
                    parsed.uv_reference =
                        PathBuf::from(args.next().context("--uv-ref requires a path")?)
                }
                "--atol" => {
                    parsed.atol = args.next().context("--atol requires a value")?.parse()?
                }
                "--mean-atol" => {
                    parsed.mean_atol = Some(
                        args.next()
                            .context("--mean-atol requires a value")?
                            .parse()?,
                    )
                }
                "--dump-dir" => {
                    parsed.dump_dir = Some(PathBuf::from(args.next().context("--dump-dir")?))
                }
                "--device" => parsed.device = args.next().context("--device")?,
                "--help" | "-h" => {
                    println!("usage: cargo run --bin source_check -- --model-dir models --f0 F0 --rand RAND --noise NOISE --ref REF --source-ref REF --noise-ref REF --uv-ref UV_REF [--dump-dir DIR] [--device cpu|metal] [--atol 1e-3] [--mean-atol 1e-7]");
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

fn write_f32_bin(path: &Path, t: &Tensor) -> Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let dims = t.dims().to_vec();
    let data = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut file =
        std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    file.write_all(&(dims.len() as u32).to_le_bytes())?;
    for dim in dims {
        file.write_all(&(dim as u32).to_le_bytes())?;
    }
    for value in data {
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(())
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
    mean_atol: Option<f64>,
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
    if let Some(mean_atol) = mean_atol {
        if mean_abs > mean_atol {
            bail!(
                "{name} FAIL: mean_abs {:.3e} exceeds tolerance {:.3e}",
                mean_abs,
                mean_atol
            );
        }
    }
    Ok(())
}

fn resolve_device(spec: &str) -> Result<Device> {
    match spec {
        "cpu" => Ok(Device::Cpu),
        "metal" => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0).context("Metal device not available")
            }
            #[cfg(not(feature = "metal"))]
            {
                bail!("--device metal requires building with --features metal")
            }
        }
        other => bail!("unknown --device {other}; expected cpu or metal"),
    }
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    let device = resolve_device(&args.device)?;
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

    let (out, sine_noise, uv) = source.forward_with_controls(&f0, Some(&rand_ini), Some(&noise))?;
    if let Some(dir) = args.dump_dir.as_deref() {
        write_f32_bin(&dir.join("source_ref_sine_merge.bin"), &out)?;
        write_f32_bin(&dir.join("source_ref_sine_noise.bin"), &sine_noise)?;
        write_f32_bin(&dir.join("source_ref_uv.bin"), &uv)?;
    }
    compare("source", &out, &ref_shape, &ref_data, args.atol, None)?;
    if let Some(path) = args.source_reference.as_deref() {
        let (shape, data) = read_f32_bin(path)?;
        compare(
            "source_ref",
            &out,
            &shape,
            &data,
            args.atol,
            args.mean_atol,
        )?;
    }
    if let Some(path) = args.noise_reference.as_deref() {
        let (shape, data) = read_f32_bin(path)?;
        compare(
            "noise_ref",
            &sine_noise,
            &shape,
            &data,
            args.atol,
            args.mean_atol,
        )?;
    }
    compare("uv", &uv, &uv_shape, &uv_data, 0.0, None)?;
    if let Some(dir) = args.dump_dir.as_deref() {
        let path = dir.join("source_ref_uv.bin");
        let (shape, data) = read_f32_bin(&path)?;
        compare("uv_ref", &uv, &shape, &data, 0.0, Some(0.0))?;
    }
    println!("OK");
    Ok(())
}
