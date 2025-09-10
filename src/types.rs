use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputSpec {
    pub batch: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub dtype: String, // "f32" | "u8" ...
}

impl InputSpec {
    pub fn validate(&self, shape: &[usize], dtype: &str) -> anyhow::Result<()> {
        anyhow::ensure!(shape.len() == 4, "Input muss 4D (NCHW) sein");
        anyhow::ensure!(shape[0] == self.batch, "Batch size passt nicht");
        anyhow::ensure!(shape[1] == self.channels, "Channels passen nicht");
        anyhow::ensure!(
            shape[2] == self.height && shape[3] == self.width,
            "H/W passen nicht"
        );
        anyhow::ensure!(dtype == self.dtype, "dtype passt nicht");
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelCfg {
    pub backend: String,
    pub device: String,
    pub model_path: String,
    #[serde(default)]
    pub gpu_ids: Vec<usize>,

    pub input_names: Vec<String>,
    pub input_shapes: Vec<Vec<usize>>,
    pub output_names: Vec<String>,
    pub output_shapes: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InputCfg {
    pub batch: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub dtype: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QueueCfg {
    pub max_batch: usize,
    pub max_wait_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RedisCfg {
    pub url: String,
    pub out_prefix: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub model: ModelCfg,
    pub input: InputCfg,
    pub queue: QueueCfg,
    pub redis: RedisCfg,
}

impl Config {
    pub fn input_spec(&self) -> InputSpec {
        InputSpec {
            batch: self.input.batch,
            channels: self.input.channels,
            height: self.input.height,
            width: self.input.width,
            dtype: self.input.dtype.clone(),
        }
    }
}

// Job/Reply

#[derive(Debug, Clone)]
pub struct Job {
    pub id: String,          // z. B. UUID
    pub tensor: ArrayD<f32>, // NCHW; kann Batch 1 sein, wird in der Mainloop gestapelt
}

#[derive(Debug, Clone)]
pub struct Batch {
    pub ids: Vec<String>,
    pub tensor: ArrayD<f32>, // NCHW; N == ids.len()
    pub actual_len: usize,
}

