use crate::types::{Batch, Job};
use anyhow::Result;
use ndarray::{ArrayD, Axis, stack};
use tokio::sync::mpsc;
use tokio::time::{self, Duration};

/// Sammelt Jobs zu einem Batch der Größe `spec_n`.
/// - Holt blockierend mindestens ein Item.
/// - Sammelt bis `max_batch` oder Timeout.
/// - Füllt ggf. mit Null-Tensoren auf, bis `spec_n` erreicht.
/// - Gibt `Batch { ids, tensor, actual_len }` zurück.
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
