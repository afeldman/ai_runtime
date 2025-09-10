use std::sync::Arc;

pub trait Preprocessor: Send + Sync {
    fn run(&self, input: ndarray::ArrayD<f32>) -> anyhow::Result<ndarray::ArrayD<f32>>;
}

pub trait Postprocessor: Send + Sync {
    fn run(&self, input: ndarray::ArrayD<f32>) -> anyhow::Result<ndarray::ArrayD<f32>>;
}

#[derive(Clone)]
pub struct Pipeline {
    pub pre: Vec<Arc<dyn Preprocessor>>,
    pub post: Vec<Arc<dyn Postprocessor>>,
}

impl Pipeline {
    pub fn new() -> Arc<Self> {
        Arc::new(Self { pre: vec![], post: vec![] })
    }

    pub fn run_pre(&self, x: ndarray::ArrayD<f32>) -> anyhow::Result<ndarray::ArrayD<f32>> {
        let mut out = x;
        for p in &self.pre {
            out = p.run(out)?;
        }
        Ok(out)
    }

    pub fn run_post(&self, x: ndarray::ArrayD<f32>) -> anyhow::Result<ndarray::ArrayD<f32>> {
        let mut out = x;
        for p in &self.post {
            out = p.run(out)?;
        }
        Ok(out)
    }
}
