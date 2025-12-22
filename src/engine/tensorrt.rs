//! NVIDIA TensorRT engine.
//!
//! This backend executes optimized inference using TensorRT on NVIDIA GPUs.
//! It loads an engine file, creates an execution context, and runs inference
//! by binding input/output buffers. Requires CUDA and TensorRT to be available.

#[cfg(feature = "tensorrt")]
use anyhow::{Result, Context};
use ndarray::{ArrayD, IxDyn};
use crate::types::Config;
use super::Engine;

/// TensorRT inference engine implementation.
pub struct TrtEngine {
    engine: tensorrt_rs::Engine,
    context: tensorrt_rs::ExecutionContext,
    device_id: i32,
    input_names: Vec<String>,
    output_names: Vec<String>,
    output_shapes: Vec<Vec<usize>>,
}

impl TrtEngine {
    /// Creates a new TensorRT engine and selects the CUDA device.
    pub fn new(cfg: &Config, device_id: Option<usize>) -> Result<Self> {
        let gpu_id = device_id.unwrap_or(0) as i32;

        unsafe {
            let res = cuda_sys::cuda::cudaSetDevice(gpu_id);
            anyhow::ensure!(res == 0, "cudaSetDevice({}) fehlgeschlagen, code={}", gpu_id, res);
        }

        let engine = tensorrt_rs::Engine::from_file(&cfg.model.model_path)
            .with_context(|| format!("TensorRT: Engine laden fehlgeschlagen: {}", cfg.model.model_path))?;
        let context = engine.create_execution_context()
            .context("TensorRT: ExecutionContext erstellen fehlgeschlagen")?;

        anyhow::ensure!(
            cfg.model.input_names.len() == cfg.model.input_shapes.len(),
            "input_names und input_shapes haben unterschiedliche Länge"
        );
        anyhow::ensure!(
            cfg.model.output_names.len() == cfg.model.output_shapes.len(),
            "output_names und output_shapes haben unterschiedliche Länge"
        );

        Ok(Self {
            engine,
            context,
            device_id: gpu_id,
            input_names: cfg.model.input_names.clone(),
            output_names: cfg.model.output_names.clone(),
            output_shapes: cfg.model.output_shapes.clone(),
        })
    }
}

impl Engine for TrtEngine {
    fn name(&self) -> &'static str { "tensorrt" }

    /// Runs inference using the TensorRT execution context and returns the output tensor.
    fn infer_array(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>> {
        unsafe {
            let res = cuda_sys::cuda::cudaSetDevice(self.device_id);
            anyhow::ensure!(res == 0, "cudaSetDevice({}) fehlgeschlagen, code={}", self.device_id, res);
        }

        let shape: Vec<i32> = input.shape().iter().map(|&d| d as i32).collect();
        let mut bindings = self.engine.allocate_bindings()?;

        let in_name = &self.input_names[0];
        bindings.set_input(in_name, input.as_slice().unwrap(), &shape)?;

        self.context.enqueue(&mut bindings)?;

        let out_name = &self.output_names[0];
        let output: Vec<f32> = bindings.get_output(out_name)?;
        let out_shape = IxDyn(&self.output_shapes[0]);

        let arr = ArrayD::from_shape_vec(out_shape, output)?;
        Ok(arr)
    }
}
