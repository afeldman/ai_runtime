//! OmniEngine CLI - Command-line interface for the inference runtime.
//!
//! This binary provides a simple CLI wrapper around the OmniEngine library.
//! Configuration is read from runtime.toml in the current directory.

use omniengine::start_runtime;

/// Main entry point for the OmniEngine CLI.
///
/// Reads configuration from runtime.toml and starts the inference runtime.
/// The runtime will process jobs from the input queue and write results to Redis.
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    start_runtime().await
}
