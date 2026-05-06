//! Validate stage 7 duration alignment against `tools/reference_alignment.py`.

use anyhow::{bail, Context, Result};
use kokoro_tts::model::alignment_from_durations;
use std::path::{Path, PathBuf};

#[derive(Debug)]
struct Args {
    durations: PathBuf,
    reference: PathBuf,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let mut parsed = Self {
            durations: PathBuf::from("tmp/reference_predict_duration_i64.bin"),
            reference: PathBuf::from("tmp/reference_alignment.bin"),
        };

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--durations" => {
                    parsed.durations =
                        PathBuf::from(args.next().context("--durations requires a path")?)
                }
                "--ref" => {
                    parsed.reference = PathBuf::from(args.next().context("--ref requires a path")?)
                }
                "--help" | "-h" => {
                    println!(
                        "usage: cargo run --bin alignment_check -- --durations I64_BIN --ref ALIGNMENT_BIN"
                    );
                    std::process::exit(0);
                }
                other => bail!("unknown argument {other}"),
            }
        }
        Ok(parsed)
    }
}

fn read_i64_bin(path: &Path) -> Result<(Vec<usize>, Vec<i64>)> {
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
    if bytes.len() != offset + nelem * 8 {
        bail!(
            "{} size mismatch: shape {:?} expects {} i64 values",
            path.display(),
            shape,
            nelem
        );
    }
    let mut data = Vec::with_capacity(nelem);
    for chunk in bytes[offset..].chunks_exact(8) {
        data.push(i64::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok((shape, data))
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

fn main() -> Result<()> {
    let args = Args::parse()?;
    let (_duration_shape, durations) = read_i64_bin(&args.durations)?;
    let (ref_shape, ref_data) = read_f32_bin(&args.reference)?;
    let alignment = alignment_from_durations(&durations)?;

    let shape = vec![alignment.len(), alignment.first().map_or(0, Vec::len)];
    if shape != ref_shape {
        bail!("shape mismatch: rust {:?} vs ref {:?}", shape, ref_shape);
    }

    let mut mismatches = 0usize;
    let mut first_mismatch = None;
    for (row_idx, row) in alignment.iter().enumerate() {
        for (col_idx, &got) in row.iter().enumerate() {
            let idx = row_idx * shape[1] + col_idx;
            let expected = ref_data[idx];
            if (got - expected).abs() != 0.0 {
                mismatches += 1;
                first_mismatch.get_or_insert((row_idx, col_idx, got, expected));
            }
        }
    }

    println!(
        "shape={:?} total_frames={} mismatches={}",
        shape,
        shape.get(1).copied().unwrap_or(0),
        mismatches
    );
    if let Some((row, col, got, expected)) = first_mismatch {
        bail!("first mismatch at ({row}, {col}): rust={got} ref={expected}");
    }
    println!("OK");
    Ok(())
}
