//! ONNX Runtime engine (CPU/GPU via CUDA) for `ort = 2.0.0-rc.10`.
//!
//! Highlights:
//! - Uses `ModelCfg` for input/output names and shapes.
//! - Optional CUDA support via feature `onnx-cuda`.
//! - Can run without a system-wide ONNX installation (`download-binaries`).
//!
//! Notes for `ort` v2:
//! - Call `ort::init().commit()?` globally before creating the first session.
//! - Use `SessionBuilder::new()` and `commit_from_file` to load the model.
//! - CUDA execution provider is registered only if `onnx-cuda` is enabled and
//!   `cfg.model.device == "gpu"`.

use anyhow::{Context, Result};
use ndarray::ArrayD;
use ort::{
    session::{builder::GraphOptimizationLevel, builder::SessionBuilder, Session},
    value::{DynValue, Tensor},
};
use crate::engine::Engine;
use crate::types::Config;
use std::sync::Mutex;

/// ONNX inference engine implementation.
pub struct OnnxEngine {
    session: Mutex<Session>,
    input_names: Vec<String>,
    output_names: Vec<String>,
    input_shapes: Vec<Vec<usize>>,
    output_shapes: Vec<Vec<usize>>,
}

impl OnnxEngine {
    /// Creates a new ONNX engine from the provided runtime configuration.
    ///
    /// The configuration must specify model path, I/O names and shapes, and
    /// device selection (CPU/GPU). If the `onnx-cuda` feature is enabled and
    /// `device` is GPU, the CUDA execution provider will be registered.
    pub fn new(cfg: &Config, _device_id: Option<usize>) -> Result<Self> {
        let mut builder = SessionBuilder::new()
            .with_context(|| "Fehler beim Erstellen des SessionBuilder")?;
        builder = builder.with_optimization_level(GraphOptimizationLevel::Level3)?;

        // CUDA-Provider optional aktivieren
        #[cfg(feature = "onnx-cuda")]
        {
            if cfg.model.device.to_lowercase() == "gpu" {
                let gpu_id = _device_id.unwrap_or(0) as i32;
                builder = builder
                    .with_execution_providers([ort::execution_providers::CUDAExecutionProvider::default().with_device_id(gpu_id)])?;
            }
        }

        let session = builder
            .commit_from_file(&cfg.model.model_path)
            .with_context(|| format!("ONNX-Modell konnte nicht geladen werden: {}", cfg.model.model_path))?;

        anyhow::ensure!(
            cfg.model.input_names.len() == cfg.model.input_shapes.len(),
            "input_names und input_shapes haben unterschiedliche Länge"
        );
        anyhow::ensure!(
            cfg.model.output_names.len() == cfg.model.output_shapes.len(),
            "output_names und output_shapes haben unterschiedliche Länge"
        );

        Ok(Self {
            session: Mutex::new(session),
            input_names: cfg.model.input_names.clone(),
            output_names: cfg.model.output_names.clone(),
            input_shapes: cfg.model.input_shapes.clone(),
            output_shapes: cfg.model.output_shapes.clone(),
        })
    }
}

impl Engine for OnnxEngine {
    fn name(&self) -> &'static str { "onnx" }

    /// Runs inference on the provided input tensor and returns the output tensor.
    fn infer_array(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>> {
        let mut session = self.session.lock().unwrap();

        let expected_in = &self.input_shapes[0];
        anyhow::ensure!(
            input.shape() == expected_in.as_slice(),
            "ONNX: Input-Shape passt nicht. Erwartet {:?}, bekommen {:?}",
            expected_in, input.shape()
        );

        let input_tensor: Tensor<f32> = Tensor::from_array(input.into_owned())?;

        let outputs = session.run(ort::inputs![
            &*self.input_names[0] => input_tensor
        ])?;

        let dyn_out: &DynValue = &outputs[&*self.output_names[0]];
        let out_view = dyn_out
            .try_extract_array()
            .map_err(|_| anyhow::anyhow!("ONNX: Output ist kein Tensor<f32>"))?;

        let expected_out = &self.output_shapes[0];
        anyhow::ensure!(
            out_view.shape() == expected_out.as_slice(),
            "ONNX: Output-Shape passt nicht. Erwartet {:?}, bekommen {:?}",
            expected_out, out_view.shape()
        );

        Ok(out_view.to_owned())
    }
}
