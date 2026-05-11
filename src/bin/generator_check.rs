//! Validate iSTFTNet Generator against `tools/reference_generator.py`.
//! Phase D har check:
//! `cargo run --bin generator_check -- --dump-har /tmp/har_after.bin --har-ref /tmp/har_before.bin`
//! `cargo run --features metal --bin generator_check -- --device metal --dump-har /tmp/har_after_metal.bin --har-ref /tmp/har_before_metal.bin`

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
    dump_har: Option<PathBuf>,
    har_reference: Option<PathBuf>,
    atol: f32,
    har_atol: f32,
    device: String,
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
            dump_har: None,
            har_reference: None,
            atol: 5e-3,
            har_atol: 1e-6,
            device: "cpu".to_string(),
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model-dir" => {
                    parsed.model_dir = PathBuf::from(args.next().context("--model-dir")?)
                }
                "--x" => parsed.x = PathBuf::from(args.next().context("--x")?),
                "--s" => parsed.s = PathBuf::from(args.next().context("--s")?),
                "--f0" => parsed.f0 = PathBuf::from(args.next().context("--f0")?),
                "--rand-ini" => parsed.rand_ini = PathBuf::from(args.next().context("--rand-ini")?),
                "--noise" => parsed.noise = PathBuf::from(args.next().context("--noise")?),
                "--ref" => parsed.reference = PathBuf::from(args.next().context("--ref")?),
                "--dump-har" => {
                    parsed.dump_har = Some(PathBuf::from(args.next().context("--dump-har")?))
                }
                "--har-ref" => {
                    parsed.har_reference = Some(PathBuf::from(args.next().context("--har-ref")?))
                }
                "--atol" => parsed.atol = args.next().context("--atol")?.parse()?,
                "--har-atol" => parsed.har_atol = args.next().context("--har-atol")?.parse()?,
                "--device" => parsed.device = args.next().context("--device")?,
                "--help" | "-h" => {
                    println!("usage: cargo run --bin generator_check -- [--model-dir DIR] [--x ...] [--s ...] [--f0 ...] [--rand-ini ...] [--noise ...] [--ref ...] [--dump-har PATH] [--har-ref PATH] [--device cpu|metal] [--atol T] [--har-atol T]");
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

fn write_f32_bin(path: &Path, t: &Tensor) -> Result<()> {
    use std::io::Write;
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

fn tensor_from(path: &Path, device: &Device) -> Result<(Vec<usize>, Tensor)> {
    let (shape, data) = read_f32_bin(path)?;
    let t = match shape.len() {
        2 => Tensor::from_vec(data, (shape[0], shape[1]), device)?,
        3 => Tensor::from_vec(data, (shape[0], shape[1], shape[2]), device)?,
        n => bail!("unsupported rank {n} in {}", path.display()),
    };
    Ok((shape, t))
}

fn compare_tensor(
    name: &str,
    actual: &Tensor,
    ref_shape: &[usize],
    ref_data: &[f32],
    atol: f32,
) -> Result<()> {
    if actual.dims() != ref_shape {
        bail!(
            "{name} shape mismatch: rust {:?} vs ref {:?}",
            actual.dims(),
            ref_shape
        );
    }
    let actual_data = actual
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut argmax = 0usize;
    for (i, (&actual, &expected)) in actual_data.iter().zip(ref_data).enumerate() {
        let delta = (actual - expected).abs();
        sum_abs += f64::from(delta);
        if delta > max_abs {
            max_abs = delta;
            argmax = i;
        }
    }
    let mean_abs = sum_abs / actual_data.len() as f64;
    println!(
        "{name} shape={:?} max_abs={:.3e} mean_abs={:.3e} argmax={} rust={:.6e} ref={:.6e}",
        actual.dims(),
        max_abs,
        mean_abs,
        argmax,
        actual_data[argmax],
        ref_data[argmax]
    );
    if max_abs > atol {
        bail!("FAIL: {name} max_abs {:.3e} exceeds {:.3e}", max_abs, atol);
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
    if args.dump_har.is_some() || args.har_reference.is_some() {
        let har = gen.harmonic_features_with_controls(&f0, Some(&rand_ini), Some(&noise))?;
        if let Some(path) = args.dump_har.as_deref() {
            write_f32_bin(path, &har)?;
        }
        if let Some(path) = args.har_reference.as_deref() {
            let (har_shape, har_data) = read_f32_bin(path)?;
            compare_tensor("har", &har, &har_shape, &har_data, args.har_atol)?;
        }
    }
    let out = gen.forward_with_controls(&x, &s, &f0, Some(&rand_ini), Some(&noise))?;

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
        bail!(
            "FAIL: max_abs {:.3e} exceeds {:.3e} (atol {:.3e} * max_ref {:.3e})",
            max_abs,
            args.atol * max_ref.max(1.0),
            args.atol,
            max_ref
        );
    }
    println!("OK");
    Ok(())
}
