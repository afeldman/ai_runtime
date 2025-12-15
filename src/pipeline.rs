use std::sync::Arc;
use anyhow::Result;
use ndarray::ArrayD;

use crate::scripting::plugins::{PythonPreprocessor, PythonPostprocessor};

pub trait Preprocessor: Send + Sync {
    fn run(&self, input: ArrayD<f32>) -> Result<ArrayD<f32>>;
}

pub trait Postprocessor: Send + Sync {
    fn run(&self, input: ArrayD<f32>) -> Result<ArrayD<f32>>;
}

#[derive(Clone)]
pub struct Pipeline {
    pub pre: Arc<dyn Preprocessor>,
    pub post: Arc<dyn Postprocessor>,
}

impl Pipeline {
    pub fn new(
        pre: Option<PythonPreprocessor>,
        post: Option<PythonPostprocessor>,
    ) -> Self {
        Self {
            pre: Arc::new(pre.unwrap_or_else(|| PythonPreprocessor::identity())),
            post: Arc::new(post.unwrap_or_else(|| PythonPostprocessor::identity())),
        }
    }

    pub fn run_pre(&self, x: ArrayD<f32>) -> Result<ArrayD<f32>> {
        self.pre.run(x)
    }

    pub fn run_post(&self, x: ArrayD<f32>) -> Result<ArrayD<f32>> {
        self.post.run(x)
    }
}
