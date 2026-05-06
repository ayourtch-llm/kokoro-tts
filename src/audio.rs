use anyhow::{bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream};
use std::collections::VecDeque;
use std::net::{SocketAddr, UdpSocket};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

const DEFAULT_DRAIN_MS: usize = 80;
const STREAM_BACKLOG_WARN_SECONDS: f64 = 30.0;
const REFERENCE_SAMPLE_RATE: u32 = 16_000;
const REFERENCE_PACKET_SAMPLES: usize = 320;

#[derive(Debug)]
struct PlaybackState {
    samples: Vec<f32>,
    index: usize,
    finished: bool,
    error: Option<String>,
}

pub fn play_samples(samples: &[f32], sample_rate: u32) -> Result<()> {
    if samples.is_empty() {
        return Ok(());
    }

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .context("no default output device available")?;
    let config = device
        .default_output_config()
        .context("querying default output config")?;
    let output_sample_rate = config.sample_rate().0;
    let mut playback = if output_sample_rate == sample_rate {
        samples.to_vec()
    } else {
        resample_linear(samples, sample_rate, output_sample_rate)
    };
    playback.extend(
        std::iter::repeat(0.0).take(output_sample_rate as usize * DEFAULT_DRAIN_MS / 1_000),
    );

    let state = Arc::new((
        Mutex::new(PlaybackState {
            samples: playback,
            index: 0,
            finished: false,
            error: None,
        }),
        Condvar::new(),
    ));

    let channels = config.channels() as usize;
    let stream_config = config.config();
    let stream = match config.sample_format() {
        SampleFormat::F32 => build_stream_f32(&device, &stream_config, channels, state.clone())?,
        SampleFormat::I16 => build_stream_i16(&device, &stream_config, channels, state.clone())?,
        SampleFormat::U16 => build_stream_u16(&device, &stream_config, channels, state.clone())?,
        other => bail!("unsupported output sample format: {other:?}"),
    };

    stream.play().context("starting output stream")?;

    let (lock, cvar) = &*state;
    let mut guard = lock.lock().expect("playback mutex poisoned");
    let deadline = Duration::from_millis(
        (playback_duration_ms(guard.samples.len(), output_sample_rate) + 500) as u64,
    );
    while !guard.finished {
        if let Some(err) = guard.error.take() {
            return Err(anyhow::anyhow!(err));
        }
        let (next_guard, timeout) = cvar
            .wait_timeout(guard, deadline)
            .expect("playback condvar poisoned");
        guard = next_guard;
        if timeout.timed_out() {
            if let Some(err) = guard.error.take() {
                return Err(anyhow::anyhow!(err));
            }
            if !guard.finished {
                bail!("audio playback timed out waiting for device consumption");
            }
        }
    }
    if let Some(err) = guard.error.take() {
        return Err(anyhow::anyhow!(err));
    }
    drop(guard);
    drop(stream);
    Ok(())
}

pub struct StreamingAudioOutput {
    output_sample_rate: u32,
    state: Arc<Mutex<StreamingState>>,
    _stream: Stream,
}

struct StreamingState {
    samples: VecDeque<f32>,
    reference: Option<ReferenceStreamState>,
    error: Option<String>,
}

pub struct ReferenceQueueReceipt {
    pub packets: usize,
    pub bytes: usize,
    pub duration_seconds: f64,
}

struct ReferenceStreamState {
    socket: UdpSocket,
    target: SocketAddr,
    samples: VecDeque<f32>,
    packet: Vec<u8>,
    packet_samples: usize,
    output_to_reference_ratio: f64,
    reference_credit: f64,
}

impl StreamingAudioOutput {
    pub fn open() -> Result<Self> {
        Self::open_with_reference(None)
    }

    pub fn open_with_reference(reference_out: Option<SocketAddr>) -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .context("no default output device available")?;
        let config = device
            .default_output_config()
            .context("querying default output config")?;
        let output_sample_rate = config.sample_rate().0;
        let reference = reference_out
            .map(|target| ReferenceStreamState::new(target, output_sample_rate))
            .transpose()?;
        let state = Arc::new(Mutex::new(StreamingState {
            samples: VecDeque::new(),
            reference,
            error: None,
        }));
        let channels = config.channels() as usize;
        let stream_config = config.config();
        let stream = match config.sample_format() {
            SampleFormat::F32 => {
                build_stream_f32_stream(&device, &stream_config, channels, state.clone())?
            }
            SampleFormat::I16 => {
                build_stream_i16_stream(&device, &stream_config, channels, state.clone())?
            }
            SampleFormat::U16 => {
                build_stream_u16_stream(&device, &stream_config, channels, state.clone())?
            }
            other => bail!("unsupported output sample format: {other:?}"),
        };
        stream.play().context("starting output stream")?;
        Ok(Self {
            output_sample_rate,
            state,
            _stream: stream,
        })
    }

    pub fn enqueue_samples(&self, samples: &[f32], input_sample_rate: u32) -> Result<()> {
        self.check_error()?;
        if samples.is_empty() {
            return Ok(());
        }
        let resampled = if input_sample_rate == self.output_sample_rate {
            samples.to_vec()
        } else {
            resample_linear(samples, input_sample_rate, self.output_sample_rate)
        };
        let mut guard = self.state.lock().expect("streaming mutex poisoned");
        guard.samples.extend(resampled);
        let queued_seconds = guard.samples.len() as f64 / self.output_sample_rate as f64;
        drop(guard);
        self.warn_if_backlogged(queued_seconds);
        self.check_error()?;
        Ok(())
    }

    pub fn enqueue_samples_with_reference(
        &self,
        samples: &[f32],
        input_sample_rate: u32,
        trailing_silence_samples: usize,
    ) -> Result<ReferenceQueueReceipt> {
        self.check_error()?;
        let mut audio = if samples.is_empty() || input_sample_rate == self.output_sample_rate {
            samples.to_vec()
        } else {
            resample_linear(samples, input_sample_rate, self.output_sample_rate)
        };
        let output_silence = trailing_silence_samples
            .saturating_mul(self.output_sample_rate as usize)
            / input_sample_rate.max(1) as usize;
        audio.extend(std::iter::repeat(0.0).take(output_silence));

        let reference = resample_linear(&audio, self.output_sample_rate, REFERENCE_SAMPLE_RATE);

        let mut guard = self.state.lock().expect("streaming mutex poisoned");
        guard.samples.extend(audio);
        let receipt = if let Some(reference_state) = guard.reference.as_mut() {
            let packets = reference.len().div_ceil(REFERENCE_PACKET_SAMPLES);
            let duration_seconds = reference.len() as f64 / REFERENCE_SAMPLE_RATE as f64;
            reference_state.samples.extend(reference);
            ReferenceQueueReceipt {
                packets,
                bytes: packets * REFERENCE_PACKET_SAMPLES * 4,
                duration_seconds,
            }
        } else {
            ReferenceQueueReceipt {
                packets: 0,
                bytes: 0,
                duration_seconds: 0.0,
            }
        };
        let queued_seconds = guard.samples.len() as f64 / self.output_sample_rate as f64;
        drop(guard);
        self.warn_if_backlogged(queued_seconds);
        self.check_error()?;
        Ok(receipt)
    }

    pub fn enqueue_silence(&self, duration_samples: usize) -> Result<()> {
        self.check_error()?;
        if duration_samples == 0 {
            return Ok(());
        }
        let mut guard = self.state.lock().expect("streaming mutex poisoned");
        guard
            .samples
            .extend(std::iter::repeat(0.0).take(duration_samples));
        let queued_seconds = guard.samples.len() as f64 / self.output_sample_rate as f64;
        drop(guard);
        self.warn_if_backlogged(queued_seconds);
        self.check_error()?;
        Ok(())
    }

    pub fn flush_queue(&self) -> Result<()> {
        self.check_error()?;
        let mut guard = self.state.lock().expect("streaming mutex poisoned");
        guard.samples.clear();
        if let Some(reference) = guard.reference.as_mut() {
            reference.flush_queue();
        }
        Ok(())
    }

    fn warn_if_backlogged(&self, queued_seconds: f64) {
        if queued_seconds >= STREAM_BACKLOG_WARN_SECONDS {
            tracing::warn!(
                queued_seconds = queued_seconds,
                backlog_limit_seconds = STREAM_BACKLOG_WARN_SECONDS,
                "audio playback queue is backing up"
            );
        }
    }

    fn check_error(&self) -> Result<()> {
        let mut guard = self.state.lock().expect("streaming mutex poisoned");
        if let Some(err) = guard.error.take() {
            return Err(anyhow::anyhow!(err));
        }
        Ok(())
    }
}

impl ReferenceStreamState {
    fn new(target: SocketAddr, output_sample_rate: u32) -> Result<Self> {
        let socket = UdpSocket::bind("0.0.0.0:0").context("binding reference UDP socket")?;
        socket
            .set_nonblocking(true)
            .context("setting reference UDP socket nonblocking")?;
        Ok(Self {
            socket,
            target,
            samples: VecDeque::new(),
            packet: vec![0; REFERENCE_PACKET_SAMPLES * 4],
            packet_samples: 0,
            output_to_reference_ratio: REFERENCE_SAMPLE_RATE as f64 / output_sample_rate as f64,
            reference_credit: 0.0,
        })
    }

    fn observe_output_frame(&mut self) -> Result<()> {
        self.reference_credit += self.output_to_reference_ratio;
        while self.reference_credit >= 1.0 {
            self.reference_credit -= 1.0;
            let Some(sample) = self.samples.pop_front() else {
                break;
            };
            self.push_packet_sample(sample)?;
        }
        if self.samples.is_empty() && self.packet_samples > 0 {
            self.flush_packet()?;
        }
        Ok(())
    }

    fn push_packet_sample(&mut self, sample: f32) -> Result<()> {
        let offset = self.packet_samples * 4;
        self.packet[offset..offset + 4].copy_from_slice(&sample.to_le_bytes());
        self.packet_samples += 1;
        if self.packet_samples == REFERENCE_PACKET_SAMPLES {
            self.flush_packet()?;
        }
        Ok(())
    }

    fn flush_packet(&mut self) -> Result<()> {
        for byte in &mut self.packet[self.packet_samples * 4..] {
            *byte = 0;
        }
        self.socket
            .send_to(&self.packet, self.target)
            .with_context(|| format!("sending reference packet to {}", self.target))?;
        self.packet.fill(0);
        self.packet_samples = 0;
        Ok(())
    }

    fn flush_queue(&mut self) {
        self.samples.clear();
        self.packet.fill(0);
        self.packet_samples = 0;
        self.reference_credit = 0.0;
    }
}

fn playback_duration_ms(sample_count: usize, sample_rate: u32) -> usize {
    if sample_rate == 0 {
        return 0;
    }
    sample_count.saturating_mul(1_000) / sample_rate as usize
}

pub fn resample_linear(samples: &[f32], input_rate: u32, output_rate: u32) -> Vec<f32> {
    if samples.is_empty() || input_rate == 0 || output_rate == 0 || input_rate == output_rate {
        return samples.to_vec();
    }
    if samples.len() == 1 {
        return vec![samples[0]];
    }

    let ratio = output_rate as f64 / input_rate as f64;
    let out_len = ((samples.len() as f64) * ratio).round().max(1.0) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 / ratio;
        let left = src_pos.floor() as usize;
        let frac = (src_pos - left as f64) as f32;
        let right = (left + 1).min(samples.len() - 1);
        let sample = samples[left] * (1.0 - frac) + samples[right] * frac;
        out.push(sample);
    }
    out
}

fn build_stream_f32(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    state: Arc<(Mutex<PlaybackState>, Condvar)>,
) -> Result<Stream> {
    let err_state = state.clone();
    let stream = device.build_output_stream(
        config,
        move |output: &mut [f32], _| fill_output_f32(output, channels, &state),
        move |err| {
            let (lock, cvar) = &*err_state;
            let mut guard = lock.lock().expect("playback mutex poisoned");
            guard.error = Some(err.to_string());
            guard.finished = true;
            cvar.notify_all();
        },
        None,
    )?;
    Ok(stream)
}

fn build_stream_f32_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    state: Arc<Mutex<StreamingState>>,
) -> Result<Stream> {
    let err_state = state.clone();
    let stream = device.build_output_stream(
        config,
        move |output: &mut [f32], _| fill_stream_output_f32(output, channels, &state),
        move |err| {
            let mut guard = err_state.lock().expect("streaming mutex poisoned");
            guard.error = Some(err.to_string());
        },
        None,
    )?;
    Ok(stream)
}

fn build_stream_i16(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    state: Arc<(Mutex<PlaybackState>, Condvar)>,
) -> Result<Stream> {
    let err_state = state.clone();
    let stream = device.build_output_stream(
        config,
        move |output: &mut [i16], _| fill_output_i16(output, channels, &state),
        move |err| {
            let (lock, cvar) = &*err_state;
            let mut guard = lock.lock().expect("playback mutex poisoned");
            guard.error = Some(err.to_string());
            guard.finished = true;
            cvar.notify_all();
        },
        None,
    )?;
    Ok(stream)
}

fn build_stream_i16_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    state: Arc<Mutex<StreamingState>>,
) -> Result<Stream> {
    let err_state = state.clone();
    let stream = device.build_output_stream(
        config,
        move |output: &mut [i16], _| fill_stream_output_i16(output, channels, &state),
        move |err| {
            let mut guard = err_state.lock().expect("streaming mutex poisoned");
            guard.error = Some(err.to_string());
        },
        None,
    )?;
    Ok(stream)
}

fn build_stream_u16(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    state: Arc<(Mutex<PlaybackState>, Condvar)>,
) -> Result<Stream> {
    let err_state = state.clone();
    let stream = device.build_output_stream(
        config,
        move |output: &mut [u16], _| fill_output_u16(output, channels, &state),
        move |err| {
            let (lock, cvar) = &*err_state;
            let mut guard = lock.lock().expect("playback mutex poisoned");
            guard.error = Some(err.to_string());
            guard.finished = true;
            cvar.notify_all();
        },
        None,
    )?;
    Ok(stream)
}

fn build_stream_u16_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    state: Arc<Mutex<StreamingState>>,
) -> Result<Stream> {
    let err_state = state.clone();
    let stream = device.build_output_stream(
        config,
        move |output: &mut [u16], _| fill_stream_output_u16(output, channels, &state),
        move |err| {
            let mut guard = err_state.lock().expect("streaming mutex poisoned");
            guard.error = Some(err.to_string());
        },
        None,
    )?;
    Ok(stream)
}

fn fill_output_f32(
    output: &mut [f32],
    channels: usize,
    state: &Arc<(Mutex<PlaybackState>, Condvar)>,
) {
    let (lock, cvar) = &**state;
    let mut guard = lock.lock().expect("playback mutex poisoned");
    let PlaybackState { samples, index, .. } = &mut *guard;
    fill_output_samples(output, channels, samples.as_slice(), index);
    if !guard.finished && guard.index >= guard.samples.len() {
        guard.finished = true;
        cvar.notify_all();
    }
}

fn fill_output_i16(
    output: &mut [i16],
    channels: usize,
    state: &Arc<(Mutex<PlaybackState>, Condvar)>,
) {
    let (lock, cvar) = &**state;
    let mut guard = lock.lock().expect("playback mutex poisoned");
    let mut frame = vec![0.0f32; channels];
    let PlaybackState { samples, index, .. } = &mut *guard;
    for chunk in output.chunks_mut(channels) {
        let sample = next_sample(samples.as_slice(), index);
        for ch in frame.iter_mut() {
            *ch = sample;
        }
        for (dst, src) in chunk.iter_mut().zip(frame.iter()) {
            *dst = float_to_i16(*src);
        }
    }
    if !guard.finished && guard.index >= guard.samples.len() {
        guard.finished = true;
        cvar.notify_all();
    }
}

fn fill_output_u16(
    output: &mut [u16],
    channels: usize,
    state: &Arc<(Mutex<PlaybackState>, Condvar)>,
) {
    let (lock, cvar) = &**state;
    let mut guard = lock.lock().expect("playback mutex poisoned");
    let mut frame = vec![0.0f32; channels];
    let PlaybackState { samples, index, .. } = &mut *guard;
    for chunk in output.chunks_mut(channels) {
        let sample = next_sample(samples.as_slice(), index);
        for ch in frame.iter_mut() {
            *ch = sample;
        }
        for (dst, src) in chunk.iter_mut().zip(frame.iter()) {
            *dst = float_to_u16(*src);
        }
    }
    if !guard.finished && guard.index >= guard.samples.len() {
        guard.finished = true;
        cvar.notify_all();
    }
}

fn fill_output_samples(output: &mut [f32], channels: usize, samples: &[f32], index: &mut usize) {
    for chunk in output.chunks_mut(channels) {
        let sample = next_sample(samples, index);
        for dst in chunk.iter_mut() {
            *dst = sample;
        }
    }
}

fn fill_stream_output_f32(output: &mut [f32], channels: usize, state: &Arc<Mutex<StreamingState>>) {
    let mut guard = state.lock().expect("streaming mutex poisoned");
    for chunk in output.chunks_mut(channels) {
        let queued_sample = guard.samples.pop_front();
        let sample = queued_sample.unwrap_or(0.0);
        if queued_sample.is_some() {
            if let Some(reference) = guard.reference.as_mut() {
                if let Err(err) = reference.observe_output_frame() {
                    guard.error = Some(err.to_string());
                }
            }
        }
        for dst in chunk.iter_mut() {
            *dst = sample;
        }
    }
}

fn fill_stream_output_i16(output: &mut [i16], channels: usize, state: &Arc<Mutex<StreamingState>>) {
    let mut guard = state.lock().expect("streaming mutex poisoned");
    for chunk in output.chunks_mut(channels) {
        let queued_sample = guard.samples.pop_front();
        let sample = queued_sample.unwrap_or(0.0);
        if queued_sample.is_some() {
            if let Some(reference) = guard.reference.as_mut() {
                if let Err(err) = reference.observe_output_frame() {
                    guard.error = Some(err.to_string());
                }
            }
        }
        let pcm = float_to_i16(sample);
        for dst in chunk.iter_mut() {
            *dst = pcm;
        }
    }
}

fn fill_stream_output_u16(output: &mut [u16], channels: usize, state: &Arc<Mutex<StreamingState>>) {
    let mut guard = state.lock().expect("streaming mutex poisoned");
    for chunk in output.chunks_mut(channels) {
        let queued_sample = guard.samples.pop_front();
        let sample = queued_sample.unwrap_or(0.0);
        if queued_sample.is_some() {
            if let Some(reference) = guard.reference.as_mut() {
                if let Err(err) = reference.observe_output_frame() {
                    guard.error = Some(err.to_string());
                }
            }
        }
        let pcm = float_to_u16(sample);
        for dst in chunk.iter_mut() {
            *dst = pcm;
        }
    }
}

fn next_sample(samples: &[f32], index: &mut usize) -> f32 {
    let sample = samples.get(*index).copied().unwrap_or(0.0);
    if *index < samples.len() {
        *index += 1;
    }
    sample
}

fn float_to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
}

fn float_to_u16(sample: f32) -> u16 {
    let scaled = sample.clamp(-1.0, 1.0) * 0.5 + 0.5;
    (scaled * u16::MAX as f32) as u16
}
