//! Long-lived UDP TTS daemon for realtime voice integration.

use anyhow::{bail, Context, Result};
use candle_core::Device;
use kokoro_tts::audio::{StreamingAudioHandle, StreamingAudioOutput};
use kokoro_tts::default_device;
use kokoro_tts::model::Kokoro;
use kokoro_tts::phonemizer::TwoTierPhonemizer;
use kokoro_tts::synthesis::{
    resolve_resource_path, soft_normalize, synthesize_text, timestamped_wav_name, write_wav,
    SILENCE_PADDING_SAMPLES,
};
use serde::Deserialize;
use std::collections::VecDeque;
use std::fs;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs, UdpSocket};
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
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
    device: String,
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
            device: "auto".to_string(),
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
                "--device" => parsed.device = args.next().context("--device")?,
                "--verbose" => parsed.verbose = true,
                "--help" | "-h" => {
                    println!(
                        "usage: cargo run --release --bin speak-server -- [--listen HOST:PORT] [--http-listen HOST:PORT] [--model-dir DIR] [--voice PATH] [--save-wav-dir DIR] [--reference-out HOST:PORT] [--speed F] [--device auto|cpu|metal] [--verbose]"
                    );
                    std::process::exit(0);
                }
                other => anyhow::bail!("unknown argument {other}"),
            }
        }
        Ok(parsed)
    }
}

fn resolve_device(spec: &str) -> Result<Device> {
    match spec {
        "auto" => Ok(default_device()),
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
        other => bail!("unknown --device {other}; expected auto, cpu, or metal"),
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .try_init()
        .ok();
    let args = Args::parse()?;
    let device = resolve_device(&args.device)?;
    tracing::info!("device: {:?}", device);
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
    let audio_output = StreamingAudioOutput::open_with_reference(reference_out)
        .context("opening output stream")?;
    let audio = audio_output.handle();
    let queue = SynthQueue::new();
    let worker_state = WorkerState {
        model,
        voice,
        audio: audio.clone(),
        save_wav_dir: args.save_wav_dir.clone(),
        speed: args.speed,
        verbose: args.verbose,
        device,
    };

    let socket = UdpSocket::bind(&args.listen)
        .with_context(|| format!("binding UDP socket on {}", args.listen))?;
    socket
        .set_nonblocking(true)
        .context("setting UDP socket nonblocking")?;
    let socket = Arc::new(socket);
    tracing::info!("listening on {}", args.listen);

    if let Some(http_listen) = args.http_listen.as_deref() {
        let http_addr = resolve_socket_addr(http_listen)?;
        let listener = TcpListener::bind(http_addr)
            .with_context(|| format!("binding HTTP calibration server on {http_addr}"))?;
        tracing::info!("http calibration listening on {}", http_addr);
        let http_queue = queue.clone();
        let http_audio = audio.clone();
        let http_socket = Arc::clone(&socket);
        thread::spawn(move || {
            if let Err(err) =
                run_http_listener(listener, http_addr, http_queue, http_audio, http_socket)
            {
                tracing::error!(%http_addr, error = %err, "http listener exited");
            }
        });
    }

    let udp_queue = queue.clone();
    let udp_socket = Arc::clone(&socket);
    let udp_audio = audio.clone();
    thread::spawn(move || {
        if let Err(err) = run_udp_listener(udp_socket, udp_queue, udp_audio) {
            tracing::error!(error = %err, "udp listener exited");
        }
    });

    let worker_queue = queue.clone();
    thread::spawn(move || run_synthesis_worker(worker_queue, worker_state));

    let _keep_audio_stream_alive = audio_output;
    loop {
        thread::park();
    }
}

fn resolve_socket_addr(spec: &str) -> Result<SocketAddr> {
    spec.to_socket_addrs()
        .with_context(|| format!("resolving {spec}"))?
        .next()
        .ok_or_else(|| anyhow::anyhow!("no socket addresses for {spec}"))
}

struct WorkerState {
    model: Arc<Kokoro>,
    voice: PathBuf,
    audio: StreamingAudioHandle,
    save_wav_dir: Option<PathBuf>,
    speed: f64,
    verbose: bool,
    device: Device,
}

#[derive(Clone)]
struct SynthQueue {
    inner: Arc<(Mutex<SynthQueueState>, Condvar)>,
}

struct SynthQueueState {
    pending: VecDeque<SynthRequest>,
    generation: u64,
}

#[derive(Debug)]
struct SynthRequest {
    text: String,
    source: SynthSource,
    generation: u64,
}

#[derive(Debug, Clone, Copy)]
enum SynthSource {
    Udp { peer: SocketAddr },
    HttpCalibrate,
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

impl SynthQueue {
    fn new() -> Self {
        Self {
            inner: Arc::new((
                Mutex::new(SynthQueueState {
                    pending: VecDeque::new(),
                    generation: 0,
                }),
                Condvar::new(),
            )),
        }
    }

    fn push(&self, text: String, source: SynthSource) {
        let (lock, cvar) = &*self.inner;
        let mut guard = lock.lock().expect("synth queue mutex poisoned");
        let generation = guard.generation;
        guard.pending.push_back(SynthRequest {
            text,
            source,
            generation,
        });
        cvar.notify_one();
    }

    fn pop(&self) -> SynthRequest {
        let (lock, cvar) = &*self.inner;
        let mut guard = lock.lock().expect("synth queue mutex poisoned");
        loop {
            if let Some(request) = guard.pending.pop_front() {
                return request;
            }
            guard = cvar.wait(guard).expect("synth queue mutex poisoned");
        }
    }

    fn flush_pending(&self) -> usize {
        let (lock, _cvar) = &*self.inner;
        let mut guard = lock.lock().expect("synth queue mutex poisoned");
        let drained = guard.pending.len();
        guard.pending.clear();
        guard.generation = guard.generation.wrapping_add(1);
        drained
    }

    fn generation(&self) -> u64 {
        let (lock, _cvar) = &*self.inner;
        lock.lock().expect("synth queue mutex poisoned").generation
    }

    fn is_current(&self, generation: u64) -> bool {
        self.generation() == generation
    }

    fn try_if_current<T>(
        &self,
        generation: u64,
        f: impl FnOnce() -> Result<T>,
    ) -> Result<Option<T>> {
        let (lock, _cvar) = &*self.inner;
        let guard = lock.lock().expect("synth queue mutex poisoned");
        if guard.generation != generation {
            return Ok(None);
        }
        let result = f()?;
        Ok(Some(result))
    }
}

fn run_udp_listener(
    socket: Arc<UdpSocket>,
    queue: SynthQueue,
    audio: StreamingAudioHandle,
) -> Result<()> {
    let mut udp_buf = [0u8; 8192];
    loop {
        match socket.recv_from(&mut udp_buf) {
            Ok((len, peer)) => {
                let text = std::str::from_utf8(&udp_buf[..len])
                    .context("decoding UTF-8 datagram")?
                    .trim_end_matches(&['\r', '\n'][..])
                    .to_string();
                if text.is_empty() {
                    tracing::info!(%peer, "skipping empty datagram");
                    continue;
                }

                if text == "[FLUSH]" {
                    let drained_requests = queue.flush_pending();
                    audio.flush_queue().context("flushing playback queue")?;
                    tracing::info!(
                        %peer,
                        drained_requests,
                        "queue flushed via UDP sentinel"
                    );
                    continue;
                }

                tracing::info!(%peer, text = %text, "received datagram");
                queue.push(text, SynthSource::Udp { peer });
            }
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(err) => return Err(err).context("receiving datagram"),
        }
    }
}

fn run_http_listener(
    listener: TcpListener,
    http_addr: SocketAddr,
    queue: SynthQueue,
    audio: StreamingAudioHandle,
    udp_socket: Arc<UdpSocket>,
) -> Result<()> {
    loop {
        match listener.accept() {
            Ok((mut stream, peer)) => {
                stream
                    .set_nonblocking(false)
                    .context("setting accepted http stream blocking")?;
                if let Err(err) = handle_http_stream(&mut stream, &queue, &audio, &udp_socket) {
                    tracing::warn!(
                        %http_addr,
                        %peer,
                        error = %err,
                        "http request failed"
                    );
                }
            }
            Err(err) => tracing::warn!(%http_addr, error = %err, "http accept failed"),
        }
    }
}

fn run_synthesis_worker(queue: SynthQueue, state: WorkerState) {
    loop {
        let request = queue.pop();
        match synthesize_phrase(&state, &request.text) {
            Ok(synthesized) => {
                if !queue.is_current(request.generation) {
                    tracing::info!(
                        text = %request.text,
                        "dropping synthesized audio invalidated by flush"
                    );
                    continue;
                }
                let receipt = match enqueue_synthesized_phrase(
                    &queue,
                    request.generation,
                    &state,
                    &synthesized,
                ) {
                    Ok(Some(receipt)) => receipt,
                    Ok(None) => {
                        tracing::info!(
                            text = %request.text,
                            "dropping synthesized audio invalidated by flush"
                        );
                        continue;
                    }
                    Err(err) => {
                        tracing::warn!(error = %err, "synthesis enqueue failed");
                        continue;
                    }
                };
                log_processed_request(&request, &receipt);
            }
            Err(err) => {
                tracing::warn!(
                    source = ?request.source,
                    text = %request.text,
                    error = %err,
                    "synthesis failed"
                );
            }
        }
    }
}

struct SynthesizedPhrase {
    samples: Vec<f32>,
    receipt: ProcessReceipt,
}

fn synthesize_phrase(state: &WorkerState, text: &str) -> Result<SynthesizedPhrase> {
    let phonemizer = TwoTierPhonemizer;
    let synth_start = std::time::Instant::now();
    let samples = synthesize_text(
        state.model.as_ref(),
        &phonemizer,
        text,
        &state.voice,
        state.speed,
        &state.device,
        state.verbose,
    )?;
    let synth_elapsed = synth_start.elapsed();

    let (samples, _scale) = soft_normalize(&samples);
    Ok(SynthesizedPhrase {
        receipt: ProcessReceipt {
            synth_ms: synth_elapsed.as_millis(),
            samples: samples.len(),
            reference_packets: 0,
            reference_bytes: 0,
            reference_duration_seconds: 0.0,
            queued_ms: samples.len() * 1_000 / 24_000,
            saved_path: None,
        },
        samples,
    })
}

fn enqueue_synthesized_phrase(
    queue: &SynthQueue,
    generation: u64,
    state: &WorkerState,
    synthesized: &SynthesizedPhrase,
) -> Result<Option<ProcessReceipt>> {
    let receipt = &synthesized.receipt;
    let saved_path = if let Some(dir) = &state.save_wav_dir {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let path = dir.join(timestamped_wav_name(std::time::SystemTime::now()));
        write_wav(&synthesized.samples, &path)?;
        Some(path)
    } else {
        None
    };

    let reference_receipt = match queue.try_if_current(generation, || {
        state
            .audio
            .enqueue_samples_with_reference(&synthesized.samples, 24_000, SILENCE_PADDING_SAMPLES)
            .context("queueing playback")
    })? {
        Some(receipt) => receipt,
        None => return Ok(None),
    };
    let updated = ProcessReceipt {
        synth_ms: receipt.synth_ms,
        samples: receipt.samples,
        reference_packets: reference_receipt.packets,
        reference_bytes: reference_receipt.bytes,
        reference_duration_seconds: reference_receipt.duration_seconds,
        queued_ms: receipt.queued_ms,
        saved_path,
    };
    log_reference_queue(&updated);
    if let Some(path) = &updated.saved_path {
        tracing::info!(saved = %path.display(), "saved wav");
    }
    Ok(Some(updated))
}

fn log_processed_request(request: &SynthRequest, receipt: &ProcessReceipt) {
    match request.source {
        SynthSource::Udp { peer } => {
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
        }
        SynthSource::HttpCalibrate => {
            tracing::info!(
                http = true,
                text = %request.text,
                synth_ms = receipt.synth_ms,
                samples = receipt.samples,
                reference_packets = receipt.reference_packets,
                reference_bytes = receipt.reference_bytes,
                reference_duration_seconds = receipt.reference_duration_seconds,
                queued_ms = receipt.queued_ms,
                "processed calibration"
            );
        }
    }
}

fn handle_http_stream(
    stream: &mut TcpStream,
    queue: &SynthQueue,
    audio: &StreamingAudioHandle,
    udp_socket: &UdpSocket,
) -> Result<()> {
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
        let drained_requests = queue.flush_pending();
        audio.flush_queue().context("flushing playback queue")?;
        let drained =
            drain_pending_datagrams(udp_socket).context("draining pending UDP datagrams")?;
        tracing::info!(
            drained_requests,
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

    queue.push(phrase, SynthSource::HttpCalibrate);
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
