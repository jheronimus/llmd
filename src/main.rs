mod fetch;
mod server;
mod sys1;
mod sys2;

use anyhow::Result;
use clap::Parser;
use server::AppState;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::net::UnixListener;
use tokio::sync::{mpsc, Mutex};
use tracing::info;

#[derive(Parser, Debug)]
#[command(author, version, about = "llmd: Dual-System LLM Daemon")]
struct Args {
    #[arg(short, long, default_value = "/tmp/llm.sock")]
    socket: PathBuf,

    #[arg(short, long, default_value = "8080")]
    port: Option<u16>,

    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    #[arg(long)]
    sys1_dir: Option<PathBuf>,

    #[arg(long)]
    sys2_model: Option<PathBuf>,

    #[arg(long, default_value_t = 0)]
    idle_timeout: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let default_models_dir = PathBuf::from(format!("{home}/.cache/llmd/models"));

    // 1. Resolve or auto-download System 1 model (Laya Multilingual ONNX)
    let sys1_dir = args.sys1_dir.unwrap_or_else(|| {
        default_models_dir.join("laya-multilingual")
    });
    if let Err(e) = fetch::ensure_sys1_model(&sys1_dir) {
        tracing::warn!("Failed to ensure System 1 model: {e}");
    }

    // 2. Resolve or auto-download System 2 model (Qwen3 0.6B GGUF)
    let sys2_path = args.sys2_model.unwrap_or_else(|| {
        default_models_dir.join(fetch::DEFAULT_QWEN_FILENAME)
    });
    let _ = fetch::ensure_sys2_model(&sys2_path);

    let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let state = Arc::new(AppState {
        inference_lock: Mutex::new(()),
        sys1: Mutex::new(None),
        sys2: Mutex::new(None),
        sys1_dir,
        sys2_path,
        last_activity: AtomicU64::new(now),
        shutdown_tx: shutdown_tx.clone(),
    });

    // Idle watcher task (if idle_timeout > 0)
    let idle_timeout = args.idle_timeout;
    let watcher_state = Arc::clone(&state);
    let watcher_tx = shutdown_tx.clone();
    if idle_timeout > 0 {
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(15)).await;
                let current_time = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let last = watcher_state.last_activity.load(Ordering::Relaxed);
                if current_time.saturating_sub(last) >= idle_timeout {
                    info!("Idle timeout of {}s reached. Initiating shutdown...", idle_timeout);
                    let _ = watcher_tx.send(()).await;
                    break;
                }
            }
        });
    }

    let app = server::create_router(state);

    if let Some(port) = args.port {
        let addr = format!("{}:{}", args.host, port);
        info!("llmd listening on http://{}", addr);
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => info!("Ctrl-C received"),
                    _ = shutdown_rx.recv() => info!("Shutdown signal received"),
                }
            })
            .await?;
    } else {
        let socket_path = args.socket.clone();
        if socket_path.exists() {
            let _ = std::fs::remove_file(&socket_path);
        }

        info!("llmd listening on Unix Domain Socket {:?}", socket_path);
        let listener = UnixListener::bind(&socket_path)?;

        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => info!("Ctrl-C received"),
                    _ = shutdown_rx.recv() => info!("Shutdown signal received"),
                }
            })
            .await?;

        let _ = std::fs::remove_file(&socket_path);
    }

    info!("llmd cleanly stopped");
    Ok(())
}
