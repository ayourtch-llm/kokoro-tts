use anyhow::Context;
use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex, RwLock};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod model;
mod phonemizer;
use model::Kokoro;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SpeakRequest {
    text: String,
    #[serde(default)]
    voice: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QueueItem {
    id: String,
    text: String,
    voice: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QueueStatus {
    speaking: bool,
    queued: usize,
    current: Option<String>,
}

#[derive(Clone)]
struct AppState {
    queue: Arc<Mutex<Vec<QueueItem>>>,
    sender: mpsc::Sender<QueueItem>,
    speaking: Arc<RwLock<bool>>,
    current_text: Arc<RwLock<Option<String>>>,
    model: Arc<Option<Kokoro>>,
    _device: Device,
}

impl AppState {
    fn new(model: Option<Kokoro>, device: Device) -> (Self, mpsc::Receiver<QueueItem>) {
        let (tx, rx) = mpsc::channel(64);
        let state = Self {
            queue: Arc::new(Mutex::new(Vec::new())),
            sender: tx,
            speaking: Arc::new(RwLock::new(false)),
            current_text: Arc::new(RwLock::new(None)),
            model: Arc::new(model),
            _device: device,
        };
        (state, rx)
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer())
        .init();

    let device = Device::Cpu;
    let model_path = PathBuf::from("./models");

    let model = if model_path.exists() {
        tracing::info!("Loading Kokoro model from {}", model_path.display());
        match Kokoro::load(&model_path, &device) {
            Ok(m) => {
                tracing::info!("Model loaded successfully");
                Some(m)
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to load model: {}. Run 'cargo run --bin download-model' first.",
                    e
                );
                None
            }
        }
    } else {
        tracing::warn!(
            "Model path '{}' not found. Run 'cargo run --bin download-model' to download the model.",
            model_path.display()
        );
        None
    };

    let (state, mut rx) = AppState::new(model, device);

    // Spawn the TTS processing task
    let processing_state = state.clone();
    tokio::spawn(async move {
        if let Err(e) = process_queue(processing_state, &mut rx).await {
            tracing::error!("Queue processor error: {}", e);
        }
    });

    let app = Router::new()
        .route("/speak", post(speak_handler))
        .route("/stop", post(stop_handler))
        .route("/status", get(status_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
        .await
        .context("Failed to bind to port 3000")?;

    tracing::info!("Server listening on http://0.0.0.0:3000");
    axum::serve(listener, app).await?;

    Ok(())
}

async fn process_queue(state: AppState, rx: &mut mpsc::Receiver<QueueItem>) -> anyhow::Result<()> {
    while let Some(item) = rx.recv().await {
        *state.speaking.write().await = true;
        *state.current_text.write().await = Some(item.text.clone());

        tracing::info!(id = %item.id, "Generating speech for: {}", item.text);

        // Generate audio with kokoro model
        if let Some(ref _model) = *state.model {
            // TODO: Full inference pipeline
            // 1. Convert text to phonemes
            // 2. Load reference voice tensor
            // 3. Run model.forward(phonemes, ref_s, speed)
            // 4. Save WAV output
            //
            // For now, simulate processing
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            tracing::info!(id = %item.id, "Audio generated (stub)");
        } else {
            tracing::warn!(id = %item.id, "Model not loaded, skipping generation");
        }

        // Remove from queue after processing
        let mut queue = state.queue.lock().await;
        queue.retain(|q| q.id != item.id);

        *state.speaking.write().await = false;
        *state.current_text.write().await = None;

        tracing::info!(id = %item.id, "Done processing");
    }

    Ok(())
}

async fn speak_handler(
    State(state): State<AppState>,
    Json(req): Json<SpeakRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    if state.model.is_none() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "Model not loaded. Run 'cargo run --bin download-model' to download the model.",
            })),
        );
    }

    let id = uuid::Uuid::new_v4().to_string();
    let item = QueueItem {
        id: id.clone(),
        text: req.text.clone(),
        voice: req.voice.clone(),
    };

    state.queue.lock().await.push(item.clone());

    match state.sender.send(item).await {
        Ok(()) => (
            StatusCode::ACCEPTED,
            Json(serde_json::json!({
                "id": id,
                "status": "queued",
            })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Failed to queue: {}", e),
            })),
        ),
    }
}

async fn stop_handler(State(state): State<AppState>) -> (StatusCode, Json<serde_json::Value>) {
    // Flush the queue
    let flushed = {
        let mut queue = state.queue.lock().await;
        let len = queue.len();
        queue.clear();
        len
    };

    // Stop current speech
    *state.speaking.write().await = false;
    *state.current_text.write().await = None;

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "stopped",
            "flushed": flushed,
        })),
    )
}

async fn status_handler(State(state): State<AppState>) -> Json<QueueStatus> {
    let queue = state.queue.lock().await;
    let speaking = *state.speaking.read().await;
    let current = state.current_text.read().await.clone();

    Json(QueueStatus {
        speaking,
        queued: queue.len(),
        current,
    })
}
