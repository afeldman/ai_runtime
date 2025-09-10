use anyhow::{Result, Context};
use serde_json::json;
use ndarray::ArrayD;
use crate::types::Config;
use super::Engine;

impl TrtEngine {
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

        // Validate I/O
        anyhow::ensure!(
            cfg.model.input_names.len() == cfg.model.input_shapes.len(),
            "input_names und input_shapes haben unterschiedliche Länge"
        );
        anyhow::ensure!(
            cfg.model.output_names.len() == cfg.model.output_shapes.len(),
            "output_names und output_shapes haben unterschiedliche Länge"
        );

        Ok(Self { engine, context, device_id: gpu_id })
    }
}

impl Engine for TrtEngine {
    fn name(&self) -> &'static str { "tensorrt" }

    fn infer_array(&self, input: ArrayD<f32>) -> Result<ArrayD<f32>> {
        unsafe {
            let res = cuda_sys::cuda::cudaSetDevice(self.device_id);
            anyhow::ensure!(res == 0, "cudaSetDevice({}) fehlgeschlagen, code={}", self.device_id, res);
        }

        let shape: Vec<i32> = input.shape().iter().map(|&d| d as i32).collect();
        let mut bindings = self.engine.allocate_bindings()?;

        // hier erster Input-Name aus Config
        let in_name = &cfg.model.input_names[0];
        bindings.set_input(in_name, input.as_slice().unwrap(), &shape)?;

        self.context.enqueue(&mut bindings)?;

        // erster Output aus Config
        let out_name = &cfg.model.output_names[0];
        let output: Vec<f32> = bindings.get_output(out_name)?;

        let out_shape = cfg.model.output_shapes[0].clone();
        let arr = ArrayD::from_shape_vec(out_shape, output)?;
        Ok(arr)
    }
}
