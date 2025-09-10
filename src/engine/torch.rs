use anyhow::{Context, Result};
use ndarray::ArrayD;
use tch::{CModule, Device as TchDevice, Tensor, kind::Kind};
use crate::types::Config;
use super::Engine;

/// TorchScript Engine
pub struct TorchEngine {
    module: CModule,
    device: TchDevice,
    input_names: Vec<String>,
    output_names: Vec<String>,
    input_shapes: Vec<Vec<usize>>,
    output_shapes: Vec<Vec<usize>>,
}

impl TorchEngine {
    pub fn new(cfg: &Config, device_id: Option<usize>) -> Result<Self> {
        // Device wählen
        let device = match cfg.model.device.to_lowercase().as_str() {
            "cpu" => TchDevice::Cpu,
            "gpu" => TchDevice::Cuda(device_id.unwrap_or(0) as i32),
            _ => TchDevice::Cpu,
        };

        // TorchScript Modell laden
        let module = CModule::load_on_device(&cfg.model.model_path, device)
            .with_context(|| format!("TorchScript: Modell laden fehlgeschlagen: {}", cfg.model.model_path))?;

        // Konsistenz-Check
        anyhow::ensure!(
            cfg.model.input_names.len() == cfg.model.input_shapes.len(),
            "Torch: input_names und input_shapes haben unterschiedliche Länge"
        );
        anyhow::ensure!(
            cfg.model.output_names.len() == cfg.model.output_shapes.len(),
            "Torch: output_names und output_shapes haben unterschiedliche Länge"
        );

        Ok(Self {
            module,
            device,
            input_names: cfg.model.input_names.clone(),
            output_names: cfg.model.output_names.clone(),
            input_shapes: cfg.model.input_shapes.clone(),
            output_shapes: cfg.model.output_shapes.clone(),
        })
    }
}

impl Engine for TorchEngine {
    fn name(&self) -> &'static str { "torch" }

    fn infer_array(&self, input: ArrayD<f32>) -> Result<ArrayD<f32>> {
        // Input-Shape validieren
        let expected = &self.input_shapes[0];
        anyhow::ensure!(
            input.shape() == expected.as_slice(),
            "Torch: Input-Shape passt nicht. Erwartet {:?}, bekommen {:?}",
            expected, input.shape()
        );

        // Input nach Tensor
        let tensor = Tensor::of_slice(input.as_slice().unwrap())
            .to_device(self.device)
            .to_kind(Kind::Float)
            .reshape(&expected.iter().map(|&d| d as i64).collect::<Vec<_>>());

        // Forward Pass
        let output = self.module.forward_ts(&[tensor])?;

        // TorchScript gibt oft Tuple zurück, aber `forward_ts` gibt Vec<Tensor>
        let out0 = output[0].to_device(TchDevice::Cpu);
        let out_vec: Vec<f32> = Vec::<f32>::from(out0);

        // Output-Shape aus Config übernehmen
        let out_shape = self.output_shapes[0].clone();
        let arr = ArrayD::from_shape_vec(out_shape, out_vec)?;
        Ok(arr)
    }
}
