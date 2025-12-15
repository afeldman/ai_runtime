//! Pipeline abstraction for pre/post-processing.
//!
//! Provides a flexible system for applying transformations before and after inference.
//! Supports custom Python-based processors or identity (no-op) processors.

use std::sync::Arc;
use anyhow::Result;
use ndarray::ArrayD;

use crate::scripting::plugins::{PythonPreprocessor, PythonPostprocessor};

/// Trait for preprocessing tensors before inference.
///
/// Implementations can perform operations like normalization, resizing, or data augmentation.
pub trait Preprocessor: Send + Sync {
    fn run(&self, input: ArrayD<f32>) -> Result<ArrayD<f32>>;
}

/// Trait for postprocessing tensors after inference.
///
/// Implementations can perform operations like softmax, NMS, or result formatting.
pub trait Postprocessor: Send + Sync {
    fn run(&self, input: ArrayD<f32>) -> Result<ArrayD<f32>>;
}

/// Complete processing pipeline with pre and post stages.
///
/// Combines preprocessing and postprocessing into a single pipeline that can be
/// shared across multiple workers.
#[derive(Clone)]
pub struct Pipeline {
    pub pre: Arc<dyn Preprocessor>,
    pub post: Arc<dyn Postprocessor>,
}

impl Pipeline {
    /// Creates a new pipeline with optional Python processors.
    ///
    /// If processors are None, identity (no-op) processors are used.
    ///
    /// # Arguments
    ///
    /// * `pre` - Optional Python preprocessor
    /// * `post` - Optional Python postprocessor
    pub fn new(
        pre: Option<PythonPreprocessor>,
        post: Option<PythonPostprocessor>,
    ) -> Self {
        Self {
            pre: Arc::new(pre.unwrap_or_else(|| PythonPreprocessor::identity())),
            post: Arc::new(post.unwrap_or_else(|| PythonPostprocessor::identity())),
        }
    }

    /// Applies preprocessing to the input tensor.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor
    ///
    /// # Returns
    ///
    /// Preprocessed tensor
    pub fn run_pre(&self, x: ArrayD<f32>) -> Result<ArrayD<f32>> {
        self.pre.run(x)
    }

    /// Applies postprocessing to the output tensor.
    ///
    /// # Arguments
    ///
    /// * `x` - Output tensor from inference
    ///
    /// # Returns
    ///
    /// Postprocessed tensor
    pub fn run_post(&self, x: ArrayD<f32>) -> Result<ArrayD<f32>> {
        self.post.run(x)
    }
}
