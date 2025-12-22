//! Python-backed pre/post-processing plugins.
//!
//! This module provides `Preprocessor` and `Postprocessor` implementations
//! that call into Python functions via PyO3 and NumPy for data exchange.
//! It is useful for rapid iteration on data transformations without
//! recompiling Rust code.

use anyhow::{Context, Result};
use ndarray::{ArrayD, IxDyn};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

use crate::pipeline::{Postprocessor, Preprocessor};

/// Python-based preprocessor calling a function from a Python module.
pub struct PythonPreprocessor {
    module: Py<PyModule>,
    func_name: String,
}

/// Python-based postprocessor calling a function from a Python module.
pub struct PythonPostprocessor {
    module: Py<PyModule>,
    func_name: String,
}

impl PythonPreprocessor {
    /// Construct from a Python module and function name (e.g. module="my_plugins", func="normalize").
    pub fn new(module: &str, func: &str) -> Result<Self> {
        Python::with_gil(|py| {
            let m = PyModule::import_bound(py, module)
                .with_context(|| format!("Konnte Python-Modul '{}' nicht importieren", module))?;
            Ok(Self { module: m.into(), func_name: func.to_string() })
        })
    }

    /// Fallback identity preprocessor that returns the input unchanged.
    pub fn identity() -> Self {
        Python::with_gil(|py| {
            let code = "def identity(x): return x";
            let m = PyModule::from_code_bound(py, code, "identity.py", "identity")
                .expect("inline identity module");
            Self { module: m.into(), func_name: "identity".to_string() }
        })
    }
}

impl PythonPostprocessor {
    /// Construct from a Python module and function name.
    pub fn new(module: &str, func: &str) -> Result<Self> {
        Python::with_gil(|py| {
            let m = PyModule::import_bound(py, module)
                .with_context(|| format!("Konnte Python-Modul '{}' nicht importieren", module))?;
            Ok(Self { module: m.into(), func_name: func.to_string() })
        })
    }

    /// Fallback identity postprocessor that returns the input unchanged.
    pub fn identity() -> Self {
        Python::with_gil(|py| {
            let code = "def identity(x): return x";
            let m = PyModule::from_code_bound(py, code, "identity.py", "identity")
                .expect("inline identity module");
            Self { module: m.into(), func_name: "identity".to_string() }
        })
    }
}

impl Preprocessor for PythonPreprocessor {
    fn run(&self, input: ArrayD<f32>) -> Result<ArrayD<f32>> {
        Python::with_gil(|py| {
            let m = self.module.bind(py);
            let func = m
                .getattr(self.func_name.as_str())
                .with_context(|| format!("Funktion '{}' nicht gefunden", self.func_name))?;

            // Robust gegen ndarray-Versionen: über NumPy konvertieren (macht ggf. Kopien)
            let np_in = PyArrayDyn::<f32>::from_owned_array_bound(py, input);
            let any = func
                .call1((np_in,))
                .with_context(|| format!("Fehler beim Aufruf '{}(...)'", self.func_name))?;

            let np_out: PyReadonlyArrayDyn<f32> = any.extract().context("Python-Rückgabe ist kein NumPy-Array")?;
            let view = np_out.as_array();
            let shape = view.shape().to_vec();
            let data: Vec<f32> = view.iter().copied().collect();
            ArrayD::from_shape_vec(IxDyn(&shape), data).context("Shape/Data konnten nicht in ArrayD gebaut werden")
        })
    }
}

impl Postprocessor for PythonPostprocessor {
    fn run(&self, input: ArrayD<f32>) -> Result<ArrayD<f32>> {
        Python::with_gil(|py| {
            let m = self.module.bind(py);
            let func = m
                .getattr(self.func_name.as_str())
                .with_context(|| format!("Funktion '{}' nicht gefunden", self.func_name))?;

            let np_in = PyArrayDyn::<f32>::from_owned_array_bound(py, input);
            let any = func
                .call1((np_in,))
                .with_context(|| format!("Fehler beim Aufruf '{}(...)'", self.func_name))?;

            let np_out: PyReadonlyArrayDyn<f32> = any.extract().context("Python-Rückgabe ist kein NumPy-Array")?;
            let view = np_out.as_array();
            let shape = view.shape().to_vec();
            let data: Vec<f32> = view.iter().copied().collect();
            ArrayD::from_shape_vec(IxDyn(&shape), data).context("Shape/Data konnten nicht in ArrayD gebaut werden")
        })
    }
}
