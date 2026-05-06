//! Long-lived UDP TTS daemon for realtime voice integration.

use anyhow::{Context, Result};
use candle_core::Device;
use kokoro_tts::audio::StreamingAudioOutput;
use kokoro_tts::model::Kokoro;
use kokoro_tts::phonemizer::TwoTierPhonemizer;
use kokoro_tts::synthesis::{
    resolve_resource_path, send_reference_audio, soft_normalize, synthesize_text,
    timestamped_wav_name, write_wav, SILENCE_PADDING_SAMPLES,
};
use std::fs;
use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};
use std::path::PathBuf;

#[derive(Debug)]
struct Args {
    listen: String,
    model_dir: PathBuf,
    voice: PathBuf,
    save_wav_dir: Option<PathBuf>,
    reference_out: Option<String>,
    speed: f64,
    verbose: bool,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let mut parsed = Self {
            listen: "127.0.0.1:9876".to_string(),
            model_dir: PathBuf::from("models"),
            voice: PathBuf::from("models/voices/af_heart.safetensors"),
            save_wav_dir: None,
            reference_out: None,
            speed: 1.0,
            verbose: false,
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--listen" => parsed.listen = args.next().context("--listen")?,
                "--model-dir" => {
                    parsed.model_dir = PathBuf::from(args.next().context("--model-dir")?)
                }
                "--voice" => parsed.voice = PathBuf::from(args.next().context("--voice")?),
                "--save-wav-dir" => {
                    parsed.save_wav_dir =
                        Some(PathBuf::from(args.next().context("--save-wav-dir")?))
                }
                "--reference-out" => {
                    parsed.reference_out = Some(args.next().context("--reference-out")?)
                }
                "--speed" => parsed.speed = args.next().context("--speed")?.parse()?,
                "--verbose" => parsed.verbose = true,
                "--help" | "-h" => {
                    println!(
                        "usage: cargo run --release --bin speak-server -- [--listen HOST:PORT] [--model-dir DIR] [--voice PATH] [--save-wav-dir DIR] [--reference-out HOST:PORT] [--speed F] [--verbose]"
                    );
                    std::process::exit(0);
                }
                other => anyhow::bail!("unknown argument {other}"),
            }
        }
        Ok(parsed)
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::try_init().ok();
    let args = Args::parse()?;
    let device = Device::Cpu;
    let model_dir = resolve_resource_path(&args.model_dir);
    let voice = resolve_resource_path(&args.voice);
    let reference_out = args
        .reference_out
        .as_deref()
        .map(resolve_socket_addr)
        .transpose()?;
    let reference_socket = if reference_out.is_some() {
        Some(UdpSocket::bind("0.0.0.0:0").context("binding reference UDP socket")?)
    } else {
        None
    };

    tracing::info!("loading model from {}", model_dir.display());
    let model = Kokoro::load(&model_dir, &device)
        .with_context(|| format!("loading Kokoro from {}", model_dir.display()))?;
    let phonemizer = TwoTierPhonemizer;
    let audio = StreamingAudioOutput::open().context("opening output stream")?;

    let socket = UdpSocket::bind(&args.listen)
        .with_context(|| format!("binding UDP socket on {}", args.listen))?;
    tracing::info!("listening on {}", args.listen);

    let mut buf = [0u8; 8192];
    loop {
        let (len, peer) = socket.recv_from(&mut buf).context("receiving datagram")?;
        let text = std::str::from_utf8(&buf[..len])
            .context("decoding UTF-8 datagram")?
            .trim_end_matches(&['\r', '\n'][..])
            .to_string();
        if text.is_empty() {
            tracing::info!(%peer, "skipping empty datagram");
            continue;
        }

        tracing::info!(%peer, text = %text, "received datagram");
        let synth_start = std::time::Instant::now();
        let samples = synthesize_text(
            &model,
            &phonemizer,
            &text,
            &voice,
            args.speed,
            &device,
            args.verbose,
        )?;
        let synth_elapsed = synth_start.elapsed();

        let (samples, _scale) = soft_normalize(&samples);
        let save_path = if let Some(dir) = &args.save_wav_dir {
            fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
            let path = dir.join(timestamped_wav_name(std::time::SystemTime::now()));
            write_wav(&samples, &path)?;
            Some(path)
        } else {
            None
        };
        audio
            .enqueue_samples(&samples, 24_000)
            .context("queueing playback")?;
        audio
            .enqueue_silence(SILENCE_PADDING_SAMPLES)
            .context("queueing inter-datagram silence")?;
        let (reference_packets, reference_bytes) =
            if let (Some(socket), Some(target)) = (&reference_socket, reference_out) {
                send_reference_audio(socket, target, &samples, SILENCE_PADDING_SAMPLES)
                    .context("sending reference audio")?
            } else {
                (0, 0)
            };

        if let Some(path) = &save_path {
            tracing::info!(
                %peer,
                synth_ms = synth_elapsed.as_millis(),
                samples = samples.len(),
                reference_packets,
                reference_bytes,
                saved = %path.display(),
                queued_ms = (samples.len() * 1_000 / 24_000),
                "processed datagram"
            );
            tracing::info!(saved = %path.display(), "saved wav");
        } else {
            tracing::info!(
                %peer,
                synth_ms = synth_elapsed.as_millis(),
                samples = samples.len(),
                reference_packets,
                reference_bytes,
                queued_ms = (samples.len() * 1_000 / 24_000),
                "processed datagram"
            );
        }
    }
}

fn resolve_socket_addr(spec: &str) -> Result<SocketAddr> {
    spec.to_socket_addrs()
        .with_context(|| format!("resolving {spec}"))?
        .next()
        .ok_or_else(|| anyhow::anyhow!("no socket addresses for {spec}"))
}
