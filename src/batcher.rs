//! Dynamic batching implementation for efficient inference.
//!
//! This module provides functionality to collect individual jobs into batches
//! with configurable size limits and timeouts. Smaller batches are padded to
//! match the model's expected batch size.

use crate::types::{Batch, Job};
use anyhow::Result;
use ndarray::{ArrayD, Axis, stack};
use tokio::sync::mpsc;
use tokio::time::{self, Duration};

/// Collects jobs into a batch of size `spec_n`.
///
/// This function implements dynamic batching by:
/// 1. Blocking to receive at least one job
/// 2. Collecting additional jobs up to `max_batch` or until timeout
/// 3. Padding with zero tensors if needed to reach `spec_n`
///
/// # Arguments
///
/// * `spec_n` - Target batch size (required by model)
/// * `rx` - Channel receiver for incoming jobs
/// * `max_batch` - Maximum number of real jobs to collect
/// * `max_wait_ms` - Maximum milliseconds to wait for additional jobs
///
/// # Returns
///
/// * `Ok(Some(Batch))` - Successfully created batch
/// * `Ok(None)` - Channel closed, no more jobs
/// * `Err(e)` - Error during batch construction
///
/// # Example
///
/// ```no_run
/// use omniengine::batcher::collect_batch;
/// use tokio::sync::mpsc;
///
/// # async fn example() {
/// let (tx, mut rx) = mpsc::channel(100);
/// let batch = collect_batch(4, &mut rx, 4, 100).await.unwrap();
/// # }
/// ```
pub async fn collect_batch(
    spec_n: usize,
    rx: &mut mpsc::Receiver<Job>,
    max_batch: usize,
    max_wait_ms: u64,
) -> Result<Option<Batch>> {
    let mut ids = Vec::with_capacity(max_batch);
    let mut items: Vec<ArrayD<f32>> = Vec::with_capacity(max_batch);

    // blockierend erstes Item holen
    let first = match rx.recv().await {
        Some(j) => j,
        None => return Ok(None),
    };
    ids.push(first.id);
    items.push(first.tensor);

    // bis max_batch sammeln, mit Timer
    let deadline = Duration::from_millis(max_wait_ms);
    let timer = time::sleep(deadline);
    tokio::pin!(timer);

    while ids.len() < max_batch {
        tokio::select! {
            biased;
            _ = &mut timer => break,
            maybe_job = rx.recv() => {
                match maybe_job {
                    Some(j) => {
                        ids.push(j.id);
                        items.push(j.tensor);
                        if ids.len() >= max_batch { break; }
                    }
                    None => break,
                }
            }
        }
    }

    let actual_len = items.len();

    // Padding bis spec_n
    while items.len() < spec_n {
        let shape = items[0].shape().to_vec();
        items.push(ArrayD::<f32>::zeros(shape));
        ids.push(format!("DUMMY-{}", items.len()));
    }

    // stapeln entlang N
    let views: Vec<_> = items.iter().map(|a| a.view()).collect();
    let batch_tensor = stack(Axis(0), &views)?;

    anyhow::ensure!(
        batch_tensor.shape()[0] == spec_n,
        "Batch-Größe {} entspricht nicht spec_n {}",
        batch_tensor.shape()[0],
        spec_n
    );

    Ok(Some(Batch { ids, tensor: batch_tensor, actual_len }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[tokio::test]
    async fn test_collect_batch_single_job() {
        let (tx, mut rx) = mpsc::channel(10);
        
        let job = Job {
            id: "job1".to_string(),
            tensor: Array::zeros((1, 3, 64, 64)).into_dyn(),
        };
        
        tx.send(job).await.unwrap();
        drop(tx);
        
        let batch = collect_batch(4, &mut rx, 4, 100).await.unwrap().unwrap();
        
        assert_eq!(batch.actual_len, 1);
        assert_eq!(batch.ids.len(), 4); // padded to spec_n
        assert_eq!(batch.tensor.shape()[0], 4);
    }

    #[tokio::test]
    async fn test_collect_batch_multiple_jobs() {
        let (tx, mut rx) = mpsc::channel(10);
        
        for i in 0..3 {
            let job = Job {
                id: format!("job{}", i),
                tensor: Array::ones((1, 3, 32, 32)).into_dyn(),
            };
            tx.send(job).await.unwrap();
        }
        drop(tx);
        
        let batch = collect_batch(4, &mut rx, 4, 100).await.unwrap().unwrap();
        
        assert_eq!(batch.actual_len, 3);
        assert_eq!(batch.ids.len(), 4);
        assert_eq!(batch.tensor.shape(), &[4, 1, 3, 32, 32]);
    }

    #[tokio::test]
    async fn test_collect_batch_channel_closed() {
        let (tx, mut rx) = mpsc::channel::<Job>(10);
        drop(tx); // close channel immediately
        
        let result = collect_batch(4, &mut rx, 4, 100).await.unwrap();
        
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_collect_batch_max_batch_limit() {
        let (tx, mut rx) = mpsc::channel(10);
        
        for i in 0..6 {
            let job = Job {
                id: format!("job{}", i),
                tensor: Array::zeros((1, 1, 16, 16)).into_dyn(),
            };
            tx.send(job).await.unwrap();
        }
        
        // max_batch is 4, so only first 4 should be collected
        let batch = collect_batch(4, &mut rx, 4, 10).await.unwrap().unwrap();
        
        assert_eq!(batch.actual_len, 4);
        assert_eq!(batch.ids.len(), 4);
    }
}
