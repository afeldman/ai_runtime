//! Python bindings for OmniEngine using PyO3.
//!
//! This module exposes a minimal Python interface for running inference
//! via the ONNX backend. It is intended for lightweight integration
//! and quick prototyping from Python without running the full runtime.
//!
//! Example (Python):
//!
//! ```python
//! import omniengine
//! import numpy as np
//!
//! # The constructor expects a path to a TOML config file
//! eng = omniengine.PyOnnxEngine("runtime.toml")
//! x = np.zeros((1, 3, 224, 224), dtype=np.float32)
//! y = eng.infer(x)
//! print(y.shape)
//! ```

use pyo3::prelude::*;
use pyo3::types::PyArrayDyn;
use ndarray::{ArrayD};
use crate::engine::{Engine, onnx::OnnxEngine};
use crate::types::Config;

/// Python wrapper around the ONNX engine.
///
/// The engine is configured using a TOML configuration file which provides
/// model paths, input/output names and shapes, and device selection.
#[pyclass]
pub struct PyOnnxEngine {
    inner: OnnxEngine,
}

#[pymethods]
impl PyOnnxEngine {
    /// Creates a new `PyOnnxEngine` from a TOML configuration file.
    ///
    /// The TOML file must contain a `[model]` section with fields such as
    /// `backend = "onnx"`, `model_path`, `input_names`, `input_shapes`,
    /// `output_names`, and `output_shapes`.
    #[new]
    pub fn new(path: String) -> PyResult<Self> {
        // Load config from TOML file
        let cfg: Config = toml::from_str(&std::fs::read_to_string(path)?)?;
        let inner = OnnxEngine::new(&cfg, None)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Runs inference on a NumPy array and returns the output as NumPy array.
    ///
    /// The input must match the configured input shape and dtype (f32).
    pub fn infer<'py>(&mut self, py: Python<'py>, input: &PyArrayDyn<f32>) -> PyResult<&'py PyArrayDyn<f32>> {
        let array: ArrayD<f32> = input.readonly().as_array().to_owned();
        let output = self.inner.infer_array(array)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(output.into_pyarray(py))
    }
}

/// Defines the `omniengine` Python module.
#[pymodule]
fn omniengine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyOnnxEngine>()?;
    Ok(())
}