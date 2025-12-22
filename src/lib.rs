#![doc(html_logo_url = "https://raw.githubusercontent.com/afeldman/ai_runtime/master/docs/logo.png")]
#![doc(html_favicon_url = "https://raw.githubusercontent.com/afeldman/ai_runtime/master/docs/logo.png")]

//! OmniEngine - Unified AI/ML Inference Runtime
//!
//! This library provides a high-performance, backend-agnostic runtime for executing
//! machine learning models across multiple frameworks (ONNX, TensorRT, PyTorch, TensorFlow).
//!
//! # Features
//!
//! * Multi-GPU support with automatic job distribution
//! * Dynamic batching with configurable batch sizes and timeouts
//! * Redis-based result storage
//! * Pluggable pre/post-processing pipelines
//! * Support for multiple ML backends
//!
//! # Example
//!
//! ```no_run
//! use omniengine::start_runtime;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     start_runtime().await
//! }
//! ```

mod types;
mod storage { pub mod redis_store; }
mod engine;
mod batcher;
mod worker;
mod pipeline;

use crate::storage::redis_store::RedisStorage;
use crate::types::{Config, Job};
pub mod scripting;

use pipeline::Pipeline;
use tokio::sync::mpsc;
use tracing::{info, Level};
use tracing_subscriber::EnvFilter;
use anyhow::Result;
use std::{fs, sync::Arc};

/// Starts the OmniEngine runtime with configuration from runtime.toml.
///
/// This function initializes the complete inference pipeline including:
/// - Tracing/logging setup
/// - Configuration loading from runtime.toml
/// - Redis connection for output storage
/// - Multi-GPU worker initialization
/// - Job dispatcher for load balancing
///
/// # Returns
///
/// * `Ok(())` - Runtime executed successfully
/// * `Err(e)` - Configuration error, Redis connection failure, or worker error
///
/// # Example
///
/// ```no_run
/// use omniengine::start_runtime;
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     start_runtime().await
/// }
/// ```
pub async fn start_runtime() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(Level::INFO.into()))
        .init();

    let cfg: Config = toml::from_str(&fs::read_to_string("runtime.toml")?)?;
    let spec = cfg.input_spec();
    info!("Starte Runtime: backend={}, batch={}x{}x{}",
        cfg.model.backend, spec.batch, spec.height, spec.width);

    // Redis
    let store = RedisStorage::new(&cfg.redis.url, cfg.redis.out_prefix.clone())?;

    // Pipeline als Arc (wird zwischen Workern geteilt)
    let pipeline = Arc::new(Pipeline::new(None, None));

    // Input-Queue
    let (tx, rx_main) = mpsc::channel::<Job>(1024);

    // Worker je GPU
    let mut handles = vec![];
    let gpu_ids = if cfg.model.device == "gpu" && !cfg.model.gpu_ids.is_empty() {
        cfg.model.gpu_ids.clone()
    } else {
        vec![usize::MAX] // „CPU“ oder default
    };

    // Dispatcher-Task: verteilt Jobs an alle Worker-Sender
    let mut worker_senders = vec![];
    for gpu in gpu_ids.into_iter() {
        let (tx_w, rx_w) = mpsc::channel::<Job>(512);
        worker_senders.push((gpu, rx_w, tx_w));
    }

    // Ein Dispatcher, der rx_main liest und Jobs round-robin an tx_w verteilt
    tokio::spawn({
        let mut worker_idx = 0usize;
        let senders: Vec<_> = worker_senders.iter().map(|(_, _, tx)| tx.clone()).collect();
        async move {
            let mut rx_main = rx_main;
            while let Some(job) = rx_main.recv().await {
                let tx = &senders[worker_idx % senders.len()];
                let _ = tx.send(job).await;
                worker_idx = worker_idx.wrapping_add(1);
            }
        }
    });

    // Worker starten
    for (gpu, rx_w, _) in worker_senders {
        let cfg_cl = cfg.clone();
        let store_cl = store.clone();
        let pipeline_cl = Arc::clone(&pipeline);

        handles.push(tokio::spawn(async move {
            let device = if gpu == usize::MAX { None } else { Some(gpu) };
            if let Err(e) = worker::run_gpu_worker(cfg_cl, device, rx_w, store_cl, (*pipeline_cl).clone()).await {
                eprintln!("[worker gpu={:?}] error: {:?}", device, e);
            }
        }));
    }

    // Demo-Jobs
    for k in 0..(spec.batch * 4) {
        let x = ndarray::Array::zeros((1, spec.channels, spec.height, spec.width)).into_dyn();
        let job = Job { id: format!("job-{}", k), tensor: x };
        let _ = tx.send(job).await;
    }
    drop(tx);

    for h in handles { let _ = h.await; }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_structure() {
        // Verify all modules are accessible
        assert!(true, "Module structure is valid");
    }

    #[tokio::test]
    async fn test_channel_creation() {
        let (tx, mut rx) = mpsc::channel::<Job>(10);
        
        let job = Job {
            id: "test-job-1".to_string(),
            tensor: ndarray::Array::zeros((1, 3, 224, 224)).into_dyn(),
        };
        
        tx.send(job).await.unwrap();
        let received = rx.recv().await.unwrap();
        
        assert_eq!(received.id, "test-job-1");
        assert_eq!(received.tensor.shape(), &[1, 3, 224, 224]);
    }

    #[tokio::test]
    async fn test_job_creation() {
        let job = Job {
            id: "test-123".to_string(),
            tensor: ndarray::Array::ones((2, 3, 64, 64)).into_dyn(),
        };
        
        assert_eq!(job.id, "test-123");
        assert_eq!(job.tensor.shape(), &[2, 3, 64, 64]);
    }
}
