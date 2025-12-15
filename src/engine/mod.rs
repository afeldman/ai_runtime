//! Unified interface for multiple ML inference backends.
//!
//! This module provides a common trait for different inference engines
//! (ONNX Runtime, TensorRT, PyTorch, TensorFlow) allowing runtime selection
//! based on configuration.

use anyhow::Result;
use crate::types::Config;

pub mod onnx;
#[cfg(feature = "tensorrt")]
pub mod tensorrt;
#[cfg(feature = "torch")]
pub mod torch;
#[cfg(feature = "tensorflow")]
pub mod tensorflow;

/// Trait for inference engine implementations.
///
/// All backends must implement this trait to provide a unified interface
/// for model inference regardless of the underlying framework.
pub trait Engine: Send + Sync {
    /// Returns the name of the engine backend.
    fn name(&self) -> &'static str;
    
    /// Performs inference on the input tensor.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor in NCHW format
    ///
    /// # Returns
    ///
    /// Output tensor from model inference
    fn infer_array(&mut self, input: ndarray::ArrayD<f32>) -> Result<ndarray::ArrayD<f32>>;
}

/// Factory for creating inference engines based on configuration.
///
/// Selects and initializes the appropriate backend based on the model configuration.
pub struct EngineFactory;

impl EngineFactory {
    /// Creates an engine instance for the specified device.
    ///
    /// # Arguments
    ///
    /// * `cfg` - Runtime configuration with model backend specification
    /// * `device_id` - Optional GPU device ID (None for CPU)
    ///
    /// # Returns
    ///
    /// * `Ok(Box<dyn Engine>)` - Initialized engine
    /// * `Err(e)` - Unsupported backend or initialization error
    pub fn create_for_device(cfg: &Config, device_id: Option<usize>) -> Result<Box<dyn Engine>> {
        match cfg.model.backend.as_str() {
            "onnx" => Ok(Box::new(crate::engine::onnx::OnnxEngine::new(cfg, device_id)?)),

            #[cfg(feature = "tensorrt")]
            "tensorrt" => Ok(Box::new(crate::engine::tensorrt::TrtEngine::new(cfg, device_id)?)),

            #[cfg(feature = "torch")]
            "torch" => Ok(Box::new(crate::engine::torch::TorchEngine::new(cfg, device_id)?)),

            #[cfg(feature = "tensorflow")]
            "tensorflow" => Ok(Box::new(crate::engine::tensorflow::TfEngine::new(cfg, device_id)?)),

            other => anyhow::bail!(
                "Backend '{}' nicht unterstützt (build mit features: onnx, tensorrt, torch)",
                other
            ),
        }
    }
}
