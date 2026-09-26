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
#[command(author, version, about = "llmd: High-performance Dual-System LLM Daemon")]
struct Args {
    #[arg(short, long, default_value = "/tmp/llm.sock")]
    socket: PathBuf,

    #[arg(short, long)]
    port: Option<u16>,

    #[arg(long)]
    sys1_dir: Option<PathBuf>,

    #[arg(long)]
    sys2_model: Option<PathBuf>,

    #[arg(long, default_value_t = 600)]
    idle_timeout: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let sys1_dir = args.sys1_dir.unwrap_or_else(|| {
        let p1 = PathBuf::from(format!("{home}/.cache/llm/models/laya-multilingual"));
        if p1.exists() {
            p1
        } else {
            PathBuf::from(format!("{home}/.cache/golaya/models/laya-multilingual"))
        }
    });

    let sys2_path = args.sys2_model.unwrap_or_else(|| {
        let p1 = PathBuf::from(format!("{home}/.cache/llm/models/Qwen3-1.7B-Q4_K_M.gguf"));
        if p1.exists() {
            p1
        } else {
            PathBuf::from(format!("{home}/.cache/gogemma/models/Qwen3-1.7B-Q4_K_M.gguf"))
        }
    });

    let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let state = Arc::new(AppState {
        sys1: Mutex::new(None),
        sys2: Mutex::new(None),
        sys1_dir,
        sys2_path,
        last_activity: AtomicU64::new(now),
        shutdown_tx: shutdown_tx.clone(),
    });

    // Idle watcher task
    let idle_timeout = args.idle_timeout;
    let watcher_state = Arc::clone(&state);
    let watcher_tx = shutdown_tx.clone();
    tokio::spawn(async move {
        if idle_timeout == 0 {
            return;
        }
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

    let app = server::create_router(state);

    if let Some(port) = args.port {
        let addr = format!("127.0.0.1:{port}");
        info!("llmd listening on TCP http://{}", addr);
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
