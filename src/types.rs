//! Type definitions for OmniEngine configuration and data structures.
//!
//! This module contains all core types used throughout the runtime including
//! configuration structs, job definitions, and batch structures.

use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

/// Specification for input tensor dimensions and data type.
///
/// Defines the expected shape and dtype for model inputs. Used for validation
/// before inference to ensure tensors match the model's requirements.
///
/// # Example
///
/// ```
/// # use omniengine::types::InputSpec;
/// let spec = InputSpec {
///     batch: 4,
///     channels: 3,
///     height: 224,
///     width: 224,
///     dtype: "f32".to_string(),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputSpec {
    pub batch: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub dtype: String, // "f32" | "u8" ...
}

impl InputSpec {
    /// Validates that a tensor matches this specification.
    ///
    /// Checks shape dimensions and data type against the spec requirements.
    ///
    /// # Arguments
    ///
    /// * `shape` - Tensor shape as slice [N, C, H, W]
    /// * `dtype` - Data type string (e.g., "f32", "u8")
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Tensor matches specification
    /// * `Err(e)` - Validation error with details
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

/// Model configuration including backend, device, and I/O specifications.
///
/// Defines which ML backend to use (onnx, tensorrt, torch, tensorflow),
/// device allocation (cpu/gpu), and model input/output specifications.
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

/// Input tensor configuration for the runtime.
///
/// Specifies the expected dimensions and data type for incoming inference requests.
#[derive(Debug, Clone, Deserialize)]
pub struct InputCfg {
    pub batch: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub dtype: String,
}

/// Queue configuration for dynamic batching.
///
/// Controls how jobs are collected into batches before inference.
#[derive(Debug, Clone, Deserialize)]
pub struct QueueCfg {
    pub max_batch: usize,
    pub max_wait_ms: u64,
}

/// Redis configuration for output storage.
///
/// Specifies connection details and key prefix for storing inference results.
#[derive(Debug, Clone, Deserialize)]
pub struct RedisCfg {
    pub url: String,
    pub out_prefix: String,
}

/// Complete runtime configuration.
///
/// Top-level configuration structure that combines all subsystem configs.
/// Typically loaded from runtime.toml.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub model: ModelCfg,
    pub input: InputCfg,
    pub queue: QueueCfg,
    pub redis: RedisCfg,
}

impl Config {
    /// Converts input configuration to InputSpec for validation.
    ///
    /// # Returns
    ///
    /// InputSpec derived from the input configuration
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

// Job/Reply structures

/// A single inference job with unique ID and input tensor.
///
/// Jobs are submitted to the runtime queue and processed in batches.
/// Each job carries a unique identifier for result tracking.
#[derive(Debug, Clone)]
pub struct Job {
    pub id: String,          // z. B. UUID
    pub tensor: ArrayD<f32>, // NCHW; kann Batch 1 sein, wird in der Mainloop gestapelt
}

/// A batch of jobs ready for inference.
///
/// Contains multiple jobs stacked into a single tensor along the batch dimension.
/// May include padding (dummy samples) to reach the required batch size.
///
/// # Fields
///
/// * `ids` - Job identifiers for all samples (including padding)
/// * `tensor` - Stacked tensor with shape [N, C, H, W]
/// * `actual_len` - Number of real jobs (excluding padding)
#[derive(Debug, Clone)]
pub struct Batch {
    pub ids: Vec<String>,
    pub tensor: ArrayD<f32>, // NCHW; N == ids.len()
    pub actual_len: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_spec_validate_success() {
        let spec = InputSpec {
            batch: 4,
            channels: 3,
            height: 224,
            width: 224,
            dtype: "f32".to_string(),
        };
        
        assert!(spec.validate(&[4, 3, 224, 224], "f32").is_ok());
    }

    #[test]
    fn test_input_spec_validate_wrong_batch() {
        let spec = InputSpec {
            batch: 4,
            channels: 3,
            height: 224,
            width: 224,
            dtype: "f32".to_string(),
        };
        
        assert!(spec.validate(&[2, 3, 224, 224], "f32").is_err());
    }

    #[test]
    fn test_input_spec_validate_wrong_dtype() {
        let spec = InputSpec {
            batch: 4,
            channels: 3,
            height: 224,
            width: 224,
            dtype: "f32".to_string(),
        };
        
        assert!(spec.validate(&[4, 3, 224, 224], "u8").is_err());
    }

    #[test]
    fn test_job_creation() {
        let job = Job {
            id: "test-123".to_string(),
            tensor: ndarray::Array::zeros((1, 3, 64, 64)).into_dyn(),
        };
        
        assert_eq!(job.id, "test-123");
        assert_eq!(job.tensor.shape(), &[1, 3, 64, 64]);
    }

    #[test]
    fn test_batch_creation() {
        let batch = Batch {
            ids: vec!["job1".to_string(), "job2".to_string()],
            tensor: ndarray::Array::zeros((2, 3, 64, 64)).into_dyn(),
            actual_len: 2,
        };
        
        assert_eq!(batch.ids.len(), 2);
        assert_eq!(batch.actual_len, 2);
        assert_eq!(batch.tensor.shape(), &[2, 3, 64, 64]);
    }
}
