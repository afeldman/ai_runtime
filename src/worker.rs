use crate::engine::EngineFactory;
use crate::pipeline::Pipeline;
use crate::storage::redis_store::RedisStorage;
use crate::types::{Batch, Config, Job};
use anyhow::Result;
use chrono::Utc;
use ndarray::Axis;
use tokio::sync::mpsc;
use tracing::info;

/// Worker, der Inferenz auf GPU oder CPU ausführt.
/// Holt Batches aus der Queue, führt Pre-/Postprocessing durch
/// und speichert Ergebnisse in Redis.
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

        // Preprocessing
        let x = pipeline.run_pre(batch.tensor.clone())?;
        spec.validate(x.shape(), "f32")?;

        // Inferenz
        let y = engine.infer_array(x)?;

        // Postprocessing
        let y = pipeline.run_post(y)?;

        // Ergebnisse in Redis speichern
        write_outputs(&store, &batch, y).await?;
    }

    Ok(())
}

/// Speichert Batch-Outputs in Redis.
/// Dummy-Samples (Padding) werden übersprungen.
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
