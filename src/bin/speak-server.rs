//! Long-lived UDP TTS daemon for realtime voice integration.

use anyhow::{bail, Context, Result};
use candle_core::Device;
use kokoro_tts::audio::StreamingAudioOutput;
use kokoro_tts::model::Kokoro;
use kokoro_tts::phonemizer::TwoTierPhonemizer;
use kokoro_tts::synthesis::{
    resolve_resource_path, soft_normalize, synthesize_text, timestamped_wav_name, write_wav,
    SILENCE_PADDING_SAMPLES,
};
use serde::Deserialize;
use std::fs;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs, UdpSocket};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug)]
struct Args {
    listen: String,
    http_listen: Option<String>,
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
            http_listen: None,
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
                "--http-listen" => parsed.http_listen = Some(args.next().context("--http-listen")?),
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
                        "usage: cargo run --release --bin speak-server -- [--listen HOST:PORT] [--http-listen HOST:PORT] [--model-dir DIR] [--voice PATH] [--save-wav-dir DIR] [--reference-out HOST:PORT] [--speed F] [--verbose]"
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
    tracing::info!("loading model from {}", model_dir.display());
    let model = Arc::new(
        Kokoro::load(&model_dir, &device)
            .with_context(|| format!("loading Kokoro from {}", model_dir.display()))?,
    );
    let audio = Arc::new(
        StreamingAudioOutput::open_with_reference(reference_out)
            .context("opening output stream")?,
    );
    let shared = Arc::new(SharedState {
        model,
        voice,
        audio,
        save_wav_dir: args.save_wav_dir.clone(),
        speed: args.speed,
        verbose: args.verbose,
    });

    let socket = UdpSocket::bind(&args.listen)
        .with_context(|| format!("binding UDP socket on {}", args.listen))?;
    tracing::info!("listening on {}", args.listen);

    let http_listener = if let Some(http_listen) = args.http_listen.as_deref() {
        let http_addr = resolve_socket_addr(http_listen)?;
        let listener = TcpListener::bind(http_addr)
            .with_context(|| format!("binding HTTP calibration server on {http_addr}"))?;
        listener
            .set_nonblocking(true)
            .context("setting HTTP listener nonblocking")?;
        tracing::info!("http calibration listening on {}", http_addr);
        Some((http_addr, listener))
    } else {
        None
    };

    socket
        .set_nonblocking(true)
        .context("setting UDP socket nonblocking")?;

    run_event_loop(&socket, http_listener.as_ref(), &shared)
}

fn resolve_socket_addr(spec: &str) -> Result<SocketAddr> {
    spec.to_socket_addrs()
        .with_context(|| format!("resolving {spec}"))?
        .next()
        .ok_or_else(|| anyhow::anyhow!("no socket addresses for {spec}"))
}

struct SharedState {
    model: Arc<Kokoro>,
    voice: PathBuf,
    audio: Arc<StreamingAudioOutput>,
    save_wav_dir: Option<PathBuf>,
    speed: f64,
    verbose: bool,
}

#[derive(Debug)]
struct ProcessReceipt {
    synth_ms: u128,
    samples: usize,
    reference_packets: usize,
    reference_bytes: usize,
    reference_duration_seconds: f64,
    queued_ms: usize,
    saved_path: Option<PathBuf>,
}

fn run_event_loop(
    udp_socket: &UdpSocket,
    http_listener: Option<&(SocketAddr, TcpListener)>,
    shared: &SharedState,
) -> Result<()> {
    let mut udp_buf = [0u8; 8192];
    loop {
        if let Some((http_addr, listener)) = http_listener {
            loop {
                match listener.accept() {
                    Ok((mut stream, peer)) => {
                        if let Err(err) = handle_http_stream(&mut stream, shared, udp_socket) {
                            tracing::warn!(
                                %http_addr,
                                %peer,
                                error = %err,
                                "http calibration request failed"
                            );
                        }
                    }
                    Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => break,
                    Err(err) => {
                        tracing::warn!(%http_addr, error = %err, "http calibration accept failed");
                        break;
                    }
                }
            }
        }

        match udp_socket.recv_from(&mut udp_buf) {
            Ok((len, peer)) => {
                let text = std::str::from_utf8(&udp_buf[..len])
                    .context("decoding UTF-8 datagram")?
                    .trim_end_matches(&['\r', '\n'][..])
                    .to_string();
                if text.is_empty() {
                    tracing::info!(%peer, "skipping empty datagram");
                    continue;
                }

                tracing::info!(%peer, text = %text, "received datagram");
                let receipt = process_phrase(shared, &text)?;
                tracing::info!(
                    %peer,
                    synth_ms = receipt.synth_ms,
                    samples = receipt.samples,
                    reference_packets = receipt.reference_packets,
                    reference_bytes = receipt.reference_bytes,
                    queued_ms = receipt.queued_ms,
                    save_path = %receipt
                        .saved_path
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default(),
                    "processed datagram"
                );
                if let Some(path) = &receipt.saved_path {
                    tracing::info!(saved = %path.display(), "saved wav");
                }
                log_reference_queue(&receipt);
            }
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(err) => return Err(err).context("receiving datagram"),
        }
    }
}

fn process_phrase(shared: &SharedState, text: &str) -> Result<ProcessReceipt> {
    let phonemizer = TwoTierPhonemizer;
    let synth_start = std::time::Instant::now();
    let samples = synthesize_text(
        shared.model.as_ref(),
        &phonemizer,
        text,
        &shared.voice,
        shared.speed,
        &Device::Cpu,
        shared.verbose,
    )?;
    let synth_elapsed = synth_start.elapsed();

    let (samples, _scale) = soft_normalize(&samples);
    let saved_path = if let Some(dir) = &shared.save_wav_dir {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let path = dir.join(timestamped_wav_name(std::time::SystemTime::now()));
        write_wav(&samples, &path)?;
        Some(path)
    } else {
        None
    };

    let reference_receipt = shared
        .audio
        .enqueue_samples_with_reference(&samples, 24_000, SILENCE_PADDING_SAMPLES)
        .context("queueing playback")?;

    Ok(ProcessReceipt {
        synth_ms: synth_elapsed.as_millis(),
        samples: samples.len(),
        reference_packets: reference_receipt.packets,
        reference_bytes: reference_receipt.bytes,
        reference_duration_seconds: reference_receipt.duration_seconds,
        queued_ms: samples.len() * 1_000 / 24_000,
        saved_path,
    })
}

fn handle_http_stream(
    stream: &mut TcpStream,
    shared: &SharedState,
    udp_socket: &UdpSocket,
) -> Result<()> {
    stream
        .set_nonblocking(false)
        .context("setting accepted http stream blocking")?;
    stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
    let mut buf = Vec::new();
    let mut header_end = None;
    let mut tmp = [0u8; 1024];
    while header_end.is_none() {
        let read = stream.read(&mut tmp).context("reading http request")?;
        if read == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..read]);
        header_end = find_header_end(&buf);
        if buf.len() > 64 * 1024 {
            bail!("http request headers too large");
        }
    }

    let header_end = header_end.ok_or_else(|| anyhow::anyhow!("invalid http request"))?;
    let (head, rest) = buf.split_at(header_end + 4);
    let head = String::from_utf8(head.to_vec()).context("decoding http headers")?;
    let mut lines = head.lines();
    let request_line = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("empty http request"))?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("");
    let mut content_length = 0usize;
    for line in lines {
        if let Some(value) = line.strip_prefix("Content-Length:") {
            content_length = value.trim().parse().context("parsing Content-Length")?;
        }
    }

    if method != "POST" {
        write_http_response(stream, 404, "not found")?;
        return Ok(());
    }

    if path == "/flush" {
        shared
            .audio
            .flush_queue()
            .context("flushing playback queue")?;
        let drained =
            drain_pending_datagrams(udp_socket).context("draining pending UDP datagrams")?;
        tracing::info!(
            drained_datagrams = drained,
            "queue flushed (drained pending datagrams from UDP socket)"
        );
        write_http_response(stream, 200, "ok")?;
        return Ok(());
    }

    if path != "/calibrate" {
        write_http_response(stream, 404, "not found")?;
        return Ok(());
    }

    let mut body = rest.to_vec();
    while body.len() < content_length {
        let mut chunk = [0u8; 1024];
        let read = stream.read(&mut chunk).context("reading http body")?;
        if read == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..read]);
    }
    body.truncate(content_length);
    let body = String::from_utf8(body).context("decoding http body")?;

    let phrase = if body.trim().is_empty() {
        "Recognition ready.".to_string()
    } else {
        let parsed: CalibrateRequest =
            serde_json::from_str(&body).with_context(|| "parsing calibrate JSON body")?;
        parsed
            .phrase
            .unwrap_or_else(|| "Recognition ready.".to_string())
    };

    let receipt = process_phrase(shared, &phrase)?;
    tracing::info!(
        http = true,
        text = %phrase,
        synth_ms = receipt.synth_ms,
        samples = receipt.samples,
        reference_packets = receipt.reference_packets,
        reference_bytes = receipt.reference_bytes,
        reference_duration_seconds = receipt.reference_duration_seconds,
        queued_ms = receipt.queued_ms,
        "processed calibration"
    );
    if let Some(path) = &receipt.saved_path {
        tracing::info!(saved = %path.display(), "saved wav");
    }
    log_reference_queue(&receipt);
    write_http_response(stream, 200, "ok")?;
    Ok(())
}

fn drain_pending_datagrams(socket: &UdpSocket) -> Result<usize> {
    let mut drained = 0usize;
    let mut buf = [0u8; 8192];
    loop {
        match socket.recv_from(&mut buf) {
            Ok((_len, _peer)) => drained += 1,
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => return Ok(drained),
            Err(err) => return Err(err).context("draining UDP socket"),
        }
    }
}

fn log_reference_queue(receipt: &ProcessReceipt) {
    if receipt.reference_packets > 0 {
        tracing::info!(
            reference_packets = receipt.reference_packets,
            reference_bytes = receipt.reference_bytes,
            duration_seconds = receipt.reference_duration_seconds,
            "rate-limited reference queued at 50 packets/s"
        );
    }
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|window| window == b"\r\n\r\n")
}

fn write_http_response(stream: &mut TcpStream, status: u16, body: &str) -> Result<()> {
    let status_text = match status {
        200 => "OK",
        404 => "Not Found",
        400 => "Bad Request",
        _ => "OK",
    };
    let response = format!(
        "HTTP/1.1 {status} {status_text}\r\nContent-Length: {}\r\nConnection: close\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(response.as_bytes())
        .context("writing http response")?;
    stream.flush().ok();
    Ok(())
}

#[derive(Debug, Deserialize)]
struct CalibrateRequest {
    phrase: Option<String>,
}
