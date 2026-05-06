//! Validate stage 9 Decoder pre-vocoder fusion against `tools/reference_decoder_fusion.py`.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use kokoro_tts::model::decoder::Decoder;
use kokoro_tts::model::Config;
use kokoro_tts::phonemizer::MILESTONE_TEST_PHONEMES;
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct Args {
    model: PathBuf,
    voice: PathBuf,
    config: PathBuf,
    phonemes: String,
    style_index: Option<usize>,
    asr: PathBuf,
    f0: PathBuf,
    n: PathBuf,
    reference: PathBuf,
    atol: f32,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let mut parsed = Self {
            model: PathBuf::from("models/model.safetensors"),
            voice: PathBuf::from("models/voices/af_heart.safetensors"),
            config: PathBuf::from("models/config.json"),
            phonemes: MILESTONE_TEST_PHONEMES.to_string(),
            style_index: None,
            asr: PathBuf::from("tmp/reference_decoder_asr.bin"),
            f0: PathBuf::from("tmp/reference_f0.bin"),
            n: PathBuf::from("tmp/reference_n.bin"),
            reference: PathBuf::from("tmp/reference_decoder_fusion.bin"),
            atol: 1e-4,
        };

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model" => {
                    parsed.model = PathBuf::from(args.next().context("--model requires a path")?)
                }
                "--voice" => {
                    parsed.voice = PathBuf::from(args.next().context("--voice requires a path")?)
                }
                "--config" => {
                    parsed.config = PathBuf::from(args.next().context("--config requires a path")?)
                }
                "--phonemes" => {
                    parsed.phonemes = args.next().context("--phonemes requires a value")?
                }
                "--style-index" => {
                    parsed.style_index = Some(
                        args.next()
                            .context("--style-index requires a value")?
                            .parse()
                            .context("parsing --style-index")?,
                    )
                }
                "--asr" => {
                    parsed.asr = PathBuf::from(args.next().context("--asr requires a path")?)
                }
                "--f0" => {
                    parsed.f0 = PathBuf::from(args.next().context("--f0 requires a path")?)
                }
                "--n" => {
                    parsed.n = PathBuf::from(args.next().context("--n requires a path")?)
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
                        "usage: cargo run --bin decoder_fusion_check -- --model PATH --voice PATH --config PATH --asr ASR_BIN --f0 F0_BIN --n N_BIN --ref OUT_BIN [--style-index N] [--atol F32]"
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

fn select_decoder_style(path: &Path, style_index: usize, device: &Device) -> Result<Tensor> {
    let tensors = candle_core::safetensors::load(path, device)
        .with_context(|| format!("loading {}", path.display()))?;
    let ref_s = tensors
        .get("ref_s")
        .with_context(|| format!("missing ref_s in {}", path.display()))?;
    let n = ref_s.dim(0)?;
    let idx = style_index.min(n.saturating_sub(1));
    let style = ref_s.narrow(0, idx, 1)?.squeeze(0)?;
    Ok(style.narrow(1, 0, 128)?)
}

fn tensor_from_bin(shape: &[usize], data: Vec<f32>, device: &Device) -> Result<Tensor> {
    if shape.len() != 3 {
        bail!("expected rank-3 tensor, got shape {shape:?}");
    }
    Ok(Tensor::from_vec(data, (shape[0], shape[1], shape[2]), device)?)
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
    let (asr_shape, asr_data) = read_f32_bin(&args.asr)?;
    let (f0_shape, f0_data) = read_f32_bin(&args.f0)?;
    let (n_shape, n_data) = read_f32_bin(&args.n)?;
    let (ref_shape, ref_data) = read_f32_bin(&args.reference)?;

    let style_index = args
        .style_index
        .unwrap_or_else(|| args.phonemes.chars().count().saturating_sub(1));
    let style = select_decoder_style(&args.voice, style_index, &device)?;
    let asr = tensor_from_bin(&asr_shape, asr_data, &device)?;
    if f0_shape.len() != 2 || n_shape.len() != 2 {
        bail!("expected rank-2 f0/n tensors, got {f0_shape:?} and {n_shape:?}");
    }
    let f0 = Tensor::from_vec(f0_data, (f0_shape[0], f0_shape[1]), &device)?;
    let n = Tensor::from_vec(n_data, (n_shape[0], n_shape[1]), &device)?;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[args.model.clone()], DType::F32, &device)
            .context("loading model safetensors")?
    };
    let decoder = Decoder::load(config.hidden_dim, config.style_dim, vb.pp("decoder"))
        .map_err(|e| anyhow::anyhow!("{e:#}"))?;
    let out = decoder
        .forward(&asr, &f0, &n, &style)
        .map_err(|e| anyhow::anyhow!("{e:#}"))?;
    compare(&out, &ref_shape, &ref_data, args.atol)?;
    println!("OK");
    Ok(())
}
