//! Worker implementation for GPU/CPU inference.
//!
//! This module contains the worker logic that processes batches on inference devices.
//! Workers handle the complete inference pipeline: batching, preprocessing, inference,
//! postprocessing, and result storage.

use crate::engine::EngineFactory;
use crate::pipeline::Pipeline;
use crate::storage::redis_store::RedisStorage;
use crate::types::{Batch, Config, Job};
use anyhow::Result;
use chrono::Utc;
use ndarray::Axis;
use tokio::sync::mpsc;
use tracing::info;

/// Runs an inference worker on a specific device (GPU or CPU).
///
/// The worker continuously processes jobs from the input channel:
/// 1. Collects jobs into batches using dynamic batching
/// 2. Applies preprocessing pipeline
/// 3. Validates input against model spec
/// 4. Runs inference on the configured backend
/// 5. Applies postprocessing pipeline
/// 6. Stores results in Redis
///
/// # Arguments
///
/// * `cfg` - Runtime configuration
/// * `device_id` - GPU ID (Some(n)) or CPU (None)
/// * `rx` - Channel receiver for incoming jobs
/// * `store` - Redis storage client
/// * `pipeline` - Pre/postprocessing pipeline
///
/// # Returns
///
/// * `Ok(())` - Worker completed successfully (channel closed)
/// * `Err(e)` - Error during initialization or processing
pub async fn run_gpu_worker(
    cfg: Config,
    device_id: Option<usize>,
    mut rx: mpsc::Receiver<Job>,
    store: RedisStorage,
    pipeline: Pipeline,
) -> Result<()> {
    let spec = cfg.input_spec();
    let mut engine = EngineFactory::create_for_device(&cfg, device_id)?;

    info!("Starte Engine: {}", engine.name());

    loop {
        let Some(batch) = crate::batcher::collect_batch(
            spec.batch,
            &mut rx,
            cfg.queue.max_batch.min(spec.batch),
            cfg.queue.max_wait_ms,
        )
        .await?
        else {
            break; // Channel geschlossen
        };

        let Batch { ids, tensor, actual_len } = batch;

        // Preprocessing
        let x = pipeline.run_pre(tensor)?;
        spec.validate(x.shape(), "f32")?;
        let y = engine.infer_array(x)?;
        let y = pipeline.run_post(y)?;

        // Batch "rekonstruieren", nur mit neuen Tensor-Werten
        let batch = Batch { ids, tensor: y.clone(), actual_len };
        write_outputs(&store, &batch, y).await?;
    }

    Ok(())
}

/// Stores batch inference outputs to Redis.
///
/// Writes each output tensor as JSON to Redis with metadata including timestamp and shape.
/// Dummy samples (padding) are automatically skipped based on `batch.actual_len`.
///
/// # Arguments
///
/// * `store` - Redis storage client
/// * `batch` - Batch containing job IDs and metadata
/// * `y` - Output tensor with shape [N, ...]
///
/// # Returns
///
/// * `Ok(())` - All outputs stored successfully
/// * `Err(e)` - Redis storage error or dimension mismatch
pub async fn write_outputs(
    store: &RedisStorage,
    batch: &Batch,
    y: ndarray::ArrayD<f32>,
) -> Result<()> {
    let n = y.shape()[0];
    anyhow::ensure!(
        n == batch.ids.len(),
        "Output/IDs Länge passt nicht: Output={}, IDs={}",
        n,
        batch.ids.len()
    );

    for (i, id) in batch.ids.iter().take(batch.actual_len).enumerate() {
        let slice = y.index_axis(Axis(0), i).to_owned();

        let payload = serde_json::json!({
            "id": id,
            "timestamp": Utc::now().to_rfc3339(),
            "shape": slice.shape(),
            "data": slice.iter().take(256).cloned().collect::<Vec<f32>>() // Beispiel: nur Top-256 Werte
        });

        store.store_json(id, &payload).await?;
        tracing::debug!("Stored output for job {}", id);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, ArrayD};

    #[test]
    fn test_batch_output_dimension_check() {
        let batch = Batch {
            ids: vec!["job1".to_string(), "job2".to_string()],
            tensor: Array::zeros((2, 3, 64, 64)).into_dyn(),
            actual_len: 2,
        };
        
        let y: ArrayD<f32> = Array::zeros((2, 10)).into_dyn();
        
        assert_eq!(y.shape()[0], batch.ids.len());
    }

    #[test]
    fn test_batch_actual_len_filtering() {
        let batch = Batch {
            ids: vec!["job1".to_string(), "DUMMY-1".to_string(), "DUMMY-2".to_string()],
            tensor: Array::zeros((3, 10)).into_dyn(),
            actual_len: 1, // only first job is real
        };
        
        let real_jobs: Vec<_> = batch.ids.iter().take(batch.actual_len).collect();
        assert_eq!(real_jobs.len(), 1);
        assert_eq!(real_jobs[0], "job1");
    }
}
