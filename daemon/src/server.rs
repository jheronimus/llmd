use crate::sys1::{self, Sys1Engine};
use crate::sys2::Sys2Engine;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, Mutex};

pub struct AppState {
    pub sys1: Mutex<Option<Sys1Engine>>,
    pub sys2: Mutex<Option<Sys2Engine>>,
    pub sys1_dir: std::path::PathBuf,
    pub sys2_path: std::path::PathBuf,
    pub last_activity: AtomicU64,
    pub shutdown_tx: mpsc::Sender<()>,
}

impl AppState {
    pub fn touch(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.last_activity.store(now, Ordering::Relaxed);
    }
}

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(handle_health))
        .route("/decide", post(handle_decide))
        .route("/generate", post(handle_generate))
        .route("/extract", post(handle_extract))
        .route("/shutdown", post(handle_shutdown))
        .with_state(state)
}

async fn handle_health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    state.touch();
    (StatusCode::OK, Json(serde_json::json!({"status": "ok"})))
}

#[derive(Debug, Deserialize)]
pub struct DecideRequest {
    pub state: serde_json::Value,
    pub questions: BTreeMap<String, sys1::Question>,
}

#[derive(Debug, Serialize)]
pub struct DecideResponse {
    pub answers: BTreeMap<String, sys1::Answer>,
    pub duration_ms: u64,
}

async fn handle_decide(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DecideRequest>,
) -> Result<Json<DecideResponse>, (StatusCode, String)> {
    state.touch();
    let start = Instant::now();

    let mut guard = state.sys1.lock().await;
    if guard.is_none() {
        let engine = Sys1Engine::new(&state.sys1_dir)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed loading Sys1: {e}")))?;
        *guard = Some(engine);
    }

    let engine = guard.as_mut().unwrap();
    let answers = engine
        .decide(&req.state, &req.questions)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Inference error: {e}")))?;

    Ok(Json(DecideResponse {
        answers,
        duration_ms: start.elapsed().as_millis() as u64,
    }))
}

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub system: Option<String>,
    pub prompt: String,
    pub grammar: Option<String>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct GenerateResponse {
    pub text: String,
    pub duration_ms: u64,
}

async fn handle_generate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, (StatusCode, String)> {
    state.touch();
    let start = Instant::now();

    let mut guard = state.sys2.lock().await;
    if guard.is_none() {
        let engine = Sys2Engine::new(&state.sys2_path)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed loading Sys2: {e}")))?;
        *guard = Some(engine);
    }

    let engine = guard.as_ref().unwrap();
    let text = engine
        .generate(
            req.system.as_deref().unwrap_or(""),
            &req.prompt,
            req.grammar.as_deref(),
            req.temperature.unwrap_or(0.2),
            req.top_p.unwrap_or(0.9),
            req.max_tokens.unwrap_or(1024),
        )
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Generation error: {e}")))?;

    Ok(Json(GenerateResponse {
        text,
        duration_ms: start.elapsed().as_millis() as u64,
    }))
}

#[derive(Debug, Deserialize)]
pub struct ExtractRequest {
    pub system: Option<String>,
    pub prompt: String,
    pub grammar: Option<String>,
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct ExtractResponse {
    pub raw_text: String,
    pub duration_ms: u64,
}

async fn handle_extract(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ExtractRequest>,
) -> Result<Json<ExtractResponse>, (StatusCode, String)> {
    state.touch();
    let start = Instant::now();

    let mut guard = state.sys2.lock().await;
    if guard.is_none() {
        let engine = Sys2Engine::new(&state.sys2_path)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed loading Sys2: {e}")))?;
        *guard = Some(engine);
    }

    let engine = guard.as_ref().unwrap();
    let raw_text = engine
        .generate(
            req.system.as_deref().unwrap_or(""),
            &req.prompt,
            req.grammar.as_deref(),
            0.1,
            0.9,
            req.max_tokens.unwrap_or(2048),
        )
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Extract error: {e}")))?;

    Ok(Json(ExtractResponse {
        raw_text,
        duration_ms: start.elapsed().as_millis() as u64,
    }))
}

async fn handle_shutdown(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let _ = state.shutdown_tx.send(()).await;
    (StatusCode::OK, Json(serde_json::json!({"status": "shutting down"})))
}
