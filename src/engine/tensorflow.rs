use anyhow::{Context, Result};
use ndarray::ArrayD;
use tensorflow::{Graph, Session, SessionOptions, SessionRunArgs, Tensor as TfTensor};
use crate::engine::Engine;
use crate::types::Config;

pub struct TfEngine {
    session: Session,
    input_names: Vec<String>,
    output_names: Vec<String>,
    input_shapes: Vec<Vec<usize>>,
    output_shapes: Vec<Vec<usize>>,
}

impl TfEngine {
    pub fn new(cfg: &Config, _device_id: Option<usize>) -> Result<Self> {
        let mut graph = Graph::new();
        let session = Session::new(&SessionOptions::new(), &graph)
            .context("TensorFlow: Session erstellen fehlgeschlagen")?;

        anyhow::ensure!(
            cfg.model.input_names.len() == cfg.model.input_shapes.len(),
            "TensorFlow: input_names und input_shapes haben unterschiedliche Länge"
        );
        anyhow::ensure!(
            cfg.model.output_names.len() == cfg.model.output_shapes.len(),
            "TensorFlow: output_names und output_shapes haben unterschiedliche Länge"
        );

        Ok(Self {
            session,
            input_names: cfg.model.input_names.clone(),
            output_names: cfg.model.output_names.clone(),
            input_shapes: cfg.model.input_shapes.clone(),
            output_shapes: cfg.model.output_shapes.clone(),
        })
    }
}

impl Engine for TfEngine {
    fn name(&self) -> &'static str { "tensorflow" }

    fn infer_array(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>> {
        let expected = &self.input_shapes[0];
        anyhow::ensure!(
            input.shape() == expected.as_slice(),
            "TensorFlow: Input-Shape passt nicht. Erwartet {:?}, bekommen {:?}",
            expected,
            input.shape()
        );

        // TensorFlow Tensor
        let dims: Vec<u64> = expected.iter().map(|&d| d as u64).collect();
        let mut tf_tensor = TfTensor::<f32>::new(&dims)?;
        tf_tensor.copy_from_slice(input.as_slice().unwrap());

        let mut args = SessionRunArgs::new();
        let in_op = self.session.graph().operation_by_name_required(&self.input_names[0])?;
        let out_op = self.session.graph().operation_by_name_required(&self.output_names[0])?;
        args.add_feed(&in_op, 0, &tf_tensor);
        let out_token = args.request_fetch(&out_op, 0);

        self.session.run(&mut args)?;
        let output: TfTensor<f32> = args.fetch(out_token)?;

        let arr = ArrayD::from_shape_vec(self.output_shapes[0].clone(), output.to_vec())?;
        Ok(arr)
    }
}
