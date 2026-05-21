//! Thin binary entrypoint — all the runner's actual code lives in
//! [`gelo_snp_runner`]. Tests live in the library (inline `#[cfg(test)]`
//! blocks) and in `tests/` (integration tests using the library's
//! public surface).

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    gelo_snp_runner::run().await
}
