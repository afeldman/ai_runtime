use anyhow::Result;
use crate::types::Config;
pub mod onnx;

pub trait Engine: Send + Sync {
    fn name(&self) -> &'static str;
    fn infer_array(&mut self, input: ndarray::ArrayD<f32>) -> Result<ndarray::ArrayD<f32>>;
}

pub struct EngineFactory;

impl EngineFactory {
    pub fn create_for_device(cfg: &Config, device_id: Option<usize>) -> Result<Box<dyn Engine>> {
        match cfg.model.backend.as_str() {
            "onnx" => {
                Ok(Box::new(crate::engine::onnx::OnnxEngine::new(cfg, device_id)?))
            }
            other => anyhow::bail!("Backend '{}' nicht unterstützt (nur 'onnx' CPU verfügbar)", other),
        }
    }
}
