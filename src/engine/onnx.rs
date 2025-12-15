//! ONNX Runtime Engine (CPU/GPU via CUDA) für `ort = 2.0.0-rc.10`
//!
//! Highlights
//! - Nutzt deine `ModelCfg` (Input-/Output-Namen & -Shapes).
//! - Optionaler CUDA-Support via Feature `onnx-cuda`.
//! - Läuft ohne native ONNX-Installation (Feature `download-binaries`).
//!
//! Hinweise zu ort v2:
//! - `ort::init().commit()?` muss global vor der ersten Session aufgerufen werden.
//! - `SessionBuilder::new()` + `commit_from_file` zum Laden.
//! - Execution Provider CUDA wird nur registriert, wenn `onnx-cuda` aktiv und
//!   `cfg.model.device == "gpu"` ist.

use anyhow::{Context, Result};
use ndarray::ArrayD;
use ort::{
    session::{builder::GraphOptimizationLevel, builder::SessionBuilder, Session},
    value::{DynValue, Tensor},
};
use crate::engine::Engine;
use crate::types::Config;
use std::sync::Mutex;

pub struct OnnxEngine {
    session: Mutex<Session>,
    input_names: Vec<String>,
    output_names: Vec<String>,
    input_shapes: Vec<Vec<usize>>,
    output_shapes: Vec<Vec<usize>>,
}

impl OnnxEngine {
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
