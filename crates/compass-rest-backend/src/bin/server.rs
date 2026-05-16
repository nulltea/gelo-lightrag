//! `compass-storage-server` — standalone untrusted storage backend
//! for `ring-oram`. Speaks the wire protocol defined in
//! `compass-rest-backend::wire`. Operates as plain (non-TEE) infra:
//! it stores opaque ciphertext, learns nothing beyond the
//! ORAM access pattern.
//!
//! Configuration via environment variables (keeps the binary tiny —
//! no clap dependency):
//!
//! - `COMPASS_LISTEN`  — bind address. Default `0.0.0.0:8080`.
//! - `COMPASS_DATA_DIR` — sled DB path. Default `./compass-data`.
//! - `RUST_LOG` — tracing filter. Default `info`.
//!
//! Per-tenant + per-index isolation is by URL namespace; persistence
//! happens behind `sled::Db::open_tree("{tenant}/{index}")`.

use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::{Context, Result};
use compass_rest_backend::{router, AppState};
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

const DEFAULT_LISTEN: &str = "0.0.0.0:8080";
const DEFAULT_DATA_DIR: &str = "./compass-data";

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let listen = std::env::var("COMPASS_LISTEN").unwrap_or_else(|_| DEFAULT_LISTEN.to_string());
    let data_dir: PathBuf =
        std::env::var("COMPASS_DATA_DIR")
            .unwrap_or_else(|_| DEFAULT_DATA_DIR.to_string())
            .into();

    tracing::info!(?data_dir, listen = %listen, "compass-storage-server starting");
    let db = sled::open(&data_dir)
        .with_context(|| format!("opening sled db at {}", data_dir.display()))?;
    let state = AppState::new(db);
    let app = router(state);

    let addr: SocketAddr = listen.parse().context("parsing COMPASS_LISTEN")?;
    let listener = TcpListener::bind(addr).await.context("binding listener")?;
    tracing::info!("compass-storage-server listening on http://{addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("axum serve loop")?;

    tracing::info!("compass-storage-server shut down cleanly");
    Ok(())
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}

async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    #[cfg(unix)]
    {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler");
        tokio::select! {
            _ = ctrl_c => tracing::info!("received SIGINT"),
            _ = term.recv() => tracing::info!("received SIGTERM"),
        }
    }
    #[cfg(not(unix))]
    {
        let _ = ctrl_c.await;
        tracing::info!("received Ctrl-C");
    }
}
