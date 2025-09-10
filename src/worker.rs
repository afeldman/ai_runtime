use crate::engine::EngineFactory;
use crate::storage::redis_store::RedisStorage;
use crate::types::{Config, Job};
use crate::pipeline::Pipeline;
use anyhow::Result;
use ndarray::ArrayD;
use tokio::sync::mpsc;
use chrono::Utc;

pub async fn run_gpu_worker(
    cfg: Config,
    device_id: Option<usize>,
    mut rx: mpsc::Receiver<Job>,
    store: RedisStorage,
    pipeline: Pipeline,   // <--- NEU
) -> Result<()> {
    let spec = cfg.input_spec();
    let mut engine = EngineFactory::create_for_device(&cfg, device_id)?;

    loop {
        let Some(batch) = crate::batcher::collect_batch(
            spec.batch,
            &mut rx,
            cfg.queue.max_batch.min(spec.batch),
            cfg.queue.max_wait_ms,
        ).await? else {
            break;
        };

        // Pre
        let x = pipeline.run_pre(batch.tensor)?;

        spec.validate(x.shape(), "f32")?;

        // Inferenz
        let y = engine.infer_array(x)?;

        // Post
        let y = pipeline.run_post(y)?;

        // Outputs -> Redis
        write_outputs(&store, &batch.ids, y).await?;
    }

    Ok(())
}


async fn write_outputs(store: &RedisStorage, ids: &[String], y: ArrayD<f32>) -> Result<()> {
    let n = y.shape()[0];
    anyhow::ensure!(n == ids.len(), "Output/IDs Länge passt nicht");

    for (i, id) in ids.iter().enumerate() {
        let slice = y.index_axis(ndarray::Axis(0), i).to_owned();

        let payload = serde_json::json!({
            "id": id,
            "timestamp": Utc::now(),          // ISO-8601, z. B. "2025-09-10T14:23:05Z"
            "shape": slice.shape(),
            "data": slice.iter().take(256).cloned().collect::<Vec<f32>>()
        });

        store.store_json(id, &payload).await?;
    }
    Ok(())
}
