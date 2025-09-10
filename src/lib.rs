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

/// Startet die Runtime (kannst du von main.rs aus aufrufen)
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
