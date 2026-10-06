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
use tracing::info;

pub struct AppState {
    pub inference_lock: Mutex<()>,
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
        .route("/v1/models", get(handle_openai_models))
        .route("/v1/chat/completions", post(handle_openai_chat))
        .route("/shutdown", post(handle_shutdown))
        .with_state(state)
}

async fn handle_health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    state.touch();
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "ok",
            "version": "0.2.0",
            "engines": {
                "sys1": "ready",
                "sys2": "ready"
            }
        })),
    )
}

#[derive(Debug, Deserialize)]
pub struct DecideRequest {
    pub state: serde_json::Value,
    pub questions: BTreeMap<String, sys1::Question>,
    #[serde(default = "default_true")]
    pub auto_escalate: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Serialize)]
pub struct DecideAnswerOut {
    pub answer: serde_json::Value,
    pub confidence: f32,
    pub act_cost: f32,
    pub escalated: bool,
}

#[derive(Debug, Serialize)]
pub struct DecideResponse {
    pub answers: BTreeMap<String, DecideAnswerOut>,
    pub duration_ms: u64,
}

async fn handle_decide(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DecideRequest>,
) -> Result<Json<DecideResponse>, (StatusCode, String)> {
    state.touch();
    let start = Instant::now();

    let _global_guard = state.inference_lock.lock().await;

    // 1. Run System 1 (Laya ONNX)
    let mut sys1_guard = state.sys1.lock().await;
    if sys1_guard.is_none() {
        let engine = Sys1Engine::new(&state.sys1_dir).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed loading Sys1: {e}"),
            )
        })?;
        *sys1_guard = Some(engine);
    }

    let sys1_engine = sys1_guard.as_mut().unwrap();
    let initial_answers = sys1_engine
        .decide(&req.state, &req.questions)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Inference error: {e}"),
            )
        })?;

    let mut out_answers = BTreeMap::new();
    let mut ambiguous_questions = Vec::new();

    for (qid, ans) in initial_answers {
        let is_ambiguous = ans.confidence < 0.65;
        if is_ambiguous && req.auto_escalate {
            ambiguous_questions.push(qid.clone());
        }
        out_answers.insert(
            qid,
            DecideAnswerOut {
                answer: ans.answer,
                confidence: ans.confidence,
                act_cost: ans.act_cost,
                escalated: false,
            },
        );
    }

    // 2. Auto-escalate ambiguous questions to System 2 if needed
    if !ambiguous_questions.is_empty() {
        info!(
            "Auto-escalating {} questions to System 2: {:?}",
            ambiguous_questions.len(),
            ambiguous_questions
        );

        let mut sys2_guard = state.sys2.lock().await;
        if sys2_guard.is_none() {
            let engine = Sys2Engine::new(&state.sys2_path).map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed loading Sys2 for escalation: {e}"),
                )
            })?;
            *sys2_guard = Some(engine);
        }

        let sys2_engine = sys2_guard.as_ref().unwrap();
        let state_str = if let Some(s) = req.state.as_str() {
            s.to_string()
        } else {
            req.state.to_string()
        };

        for qid in ambiguous_questions {
            if let Some(q) = req.questions.get(&qid) {
                let prompt = format!(
                    "Текст: {}\n\nВопрос: {}\nОтветь строго 'Да' или 'Нет'.",
                    state_str, q.instructions
                );
                if let Ok(sys2_text) = sys2_engine
                    .generate(
                        "Ты беспристрастный бинарный классификатор. Отвечай только 'Да' или 'Нет'.",
                        &prompt,
                        None,
                        0.1,
                        0.9,
                        16,
                    )
                    .await
                {
                    let trimmed = sys2_text.trim().to_lowercase();
                    let val = trimmed.contains("да") || trimmed.contains("yes");
                    out_answers.insert(
                        qid,
                        DecideAnswerOut {
                            answer: serde_json::Value::Bool(val),
                            confidence: 0.90,
                            act_cost: 1.0,
                            escalated: true,
                        },
                    );
                }
            }
        }
    }

    Ok(Json(DecideResponse {
        answers: out_answers,
        duration_ms: start.elapsed().as_millis() as u64,
    }))
}

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub system: Option<String>,
    pub prompt: String,
    pub grammar: Option<String>,
    pub json_schema: Option<serde_json::Value>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct GenerateResponse {
    pub text: String,
    pub duration_ms: u64,
}

fn resolve_grammar(
    grammar: Option<&str>,
    json_schema: Option<&serde_json::Value>,
) -> Result<Option<String>, (StatusCode, String)> {
    if let Some(g) = grammar {
        let trimmed = g.trim();
        if !trimmed.is_empty() {
            return Ok(Some(trimmed.to_string()));
        }
    }
    if let Some(schema_val) = json_schema {
        let schema_str = match schema_val {
            serde_json::Value::String(s) => s.clone(),
            other => serde_json::to_string(other).map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    format!("Invalid JSON schema value: {e}"),
                )
            })?,
        };
        let trimmed = schema_str.trim();
        if !trimmed.is_empty() {
            let g = llama_cpp_2::json_schema_to_grammar(trimmed).map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    format!("Failed to convert JSON schema to grammar: {e}"),
                )
            })?;
            return Ok(Some(g));
        }
    }
    Ok(None)
}

async fn handle_generate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, (StatusCode, String)> {
    state.touch();
    let start = Instant::now();

    let grammar = resolve_grammar(req.grammar.as_deref(), req.json_schema.as_ref())?;

    let _global_guard = state.inference_lock.lock().await;

    let mut guard = state.sys2.lock().await;
    if guard.is_none() {
        let engine = Sys2Engine::new(&state.sys2_path).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed loading Sys2: {e}"),
            )
        })?;
        *guard = Some(engine);
    }

    let engine = guard.as_ref().unwrap();
    let text = engine
        .generate(
            req.system.as_deref().unwrap_or(""),
            &req.prompt,
            grammar.as_deref(),
            req.temperature.unwrap_or(0.2),
            req.top_p.unwrap_or(0.9),
            req.max_tokens.unwrap_or(1024),
        )
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Generation error: {e}"),
            )
        })?;

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
    pub json_schema: Option<serde_json::Value>,
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

    let grammar = resolve_grammar(req.grammar.as_deref(), req.json_schema.as_ref())?;

    let _global_guard = state.inference_lock.lock().await;

    let mut guard = state.sys2.lock().await;
    if guard.is_none() {
        let engine = Sys2Engine::new(&state.sys2_path).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed loading Sys2: {e}"),
            )
        })?;
        *guard = Some(engine);
    }

    let engine = guard.as_ref().unwrap();
    let raw_text = engine
        .generate(
            req.system.as_deref().unwrap_or(""),
            &req.prompt,
            grammar.as_deref(),
            0.1,
            0.9,
            req.max_tokens.unwrap_or(2048),
        )
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Extract error: {e}"),
            )
        })?;

    Ok(Json(ExtractResponse {
        raw_text,
        duration_ms: start.elapsed().as_millis() as u64,
    }))
}

// ----------------- OpenAI Compatible Endpoints -----------------

async fn handle_openai_models() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "object": "list",
            "data": [
                {
                    "id": "qwen3.5-0.8b",
                    "object": "model",
                    "created": 1727700000,
                    "owned_by": "llmd"
                }
            ]
        })),
    )
}

#[derive(Debug, Deserialize)]
pub struct OpenAIChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIChatRequest {
    #[allow(dead_code)]
    pub model: Option<String>,
    pub messages: Vec<OpenAIChatMessage>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<usize>,
}

async fn handle_openai_chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OpenAIChatRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    state.touch();

    let mut system_prompt = String::new();
    let mut user_prompt = String::new();

    for msg in req.messages {
        if msg.role == "system" {
            system_prompt = msg.content;
        } else if msg.role == "user" {
            user_prompt = msg.content;
        }
    }

    let _global_guard = state.inference_lock.lock().await;

    let mut guard = state.sys2.lock().await;
    if guard.is_none() {
        let engine = Sys2Engine::new(&state.sys2_path).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed loading Sys2: {e}"),
            )
        })?;
        *guard = Some(engine);
    }

    let engine = guard.as_ref().unwrap();
    let text = engine
        .generate(
            &system_prompt,
            &user_prompt,
            None,
            req.temperature.unwrap_or(0.2),
            req.top_p.unwrap_or(0.9),
            req.max_tokens.unwrap_or(1024),
        )
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Chat completion error: {e}"),
            )
        })?;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    Ok((
        StatusCode::OK,
        Json(serde_json::json!({
            "id": format!("chatcmpl-{}", now),
            "object": "chat.completion",
            "created": now,
            "model": "qwen3.5-0.8b",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        })),
    ))
}

async fn handle_shutdown(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let _ = state.shutdown_tx.send(()).await;
    (
        StatusCode::OK,
        Json(serde_json::json!({"status": "shutting down"})),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_grammar_from_schema_string() {
        let schema =
            serde_json::json!(r#"{"type":"object","properties":{"score":{"type":"number"}}}"#);
        let g = resolve_grammar(None, Some(&schema)).unwrap();
        assert!(g.is_some());
        assert!(g.unwrap().contains("root ::="));
    }

    #[test]
    fn test_resolve_grammar_from_schema_object() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "score": { "type": "number" },
                "fit_reason": { "type": "string" }
            },
            "required": ["score", "fit_reason"]
        });
        let g = resolve_grammar(None, Some(&schema)).unwrap();
        assert!(g.is_some());
        assert!(g.unwrap().contains("root ::="));
    }

    #[test]
    fn test_resolve_grammar_precedence() {
        let grammar = "root ::= \"hello\"";
        let schema = serde_json::json!({"type": "string"});
        let g = resolve_grammar(Some(grammar), Some(&schema)).unwrap();
        assert_eq!(g.as_deref(), Some(grammar));
    }
}
