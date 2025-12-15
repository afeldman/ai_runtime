use pyo3::prelude::*;
use pyo3::types::PyArrayDyn;
use ndarray::{ArrayD};
use crate::engine::{Engine, onnx::OnnxEngine};
use crate::types::Config;

/// Python-Wrapper um eine ONNX Engine
#[pyclass]
pub struct PyOnnxEngine {
    inner: OnnxEngine,
}

#[pymethods]
impl PyOnnxEngine {
    #[new]
    pub fn new(path: String) -> PyResult<Self> {
        // Config minimal bauen (oder später per TOML laden)
        let cfg: Config = toml::from_str(&std::fs::read_to_string(path)?)?;
        let inner = OnnxEngine::new(&cfg, None)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Inferenz auf einem NumPy-Array
    pub fn infer<'py>(&mut self, py: Python<'py>, input: &PyArrayDyn<f32>) -> PyResult<&'py PyArrayDyn<f32>> {
        let array: ArrayD<f32> = input.readonly().as_array().to_owned();
        let output = self.inner.infer_array(array)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(output.into_pyarray(py))
    }
}

/// Python-Modul definieren
#[pymodule]
fn omniengine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyOnnxEngine>()?;
    Ok(())
}