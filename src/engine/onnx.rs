//! ONNX Runtime Engine (CPU) für `ort = 2.0.0-rc.10`
//!
//! Highlights
//! - Nutzt deine `ModelCfg` (Input-/Output-Namen & -Shapes).
//! - Validiert Shapes zur Laufzeit.
//! - Läuft ohne native C-Installationen (Feature `download-binaries`).
//!
//! Hinweise zu ort v2:
//! - `Environment` wurde entfernt → stattdessen `ort::init().commit()?` einmalig zu Programmstart.
//! - `Session::builder()` + `commit_from_file` statt `with_model_from_file`.
//! - Inputs via `ort::inputs![ ... ]` (auch mit Namen).
//! - Outputs sind `DynValue`s → `try_extract_array::<f32>()?`.

use anyhow::{Context, Result};
use ndarray::{ArrayD};
use ort::{
    session::{builder::GraphOptimizationLevel, builder::SessionBuilder, Session},
    value::{DynValue, Tensor}, // Tensor<T> zum Erstellen aus ndarray
};
use crate::engine::Engine;
use crate::types::Config;
use std::sync::Mutex;

/// ONNX Runtime Engine Wrapper
pub struct OnnxEngine {
    session: Mutex<Session>,
    input_names: Vec<String>,
    output_names: Vec<String>,
    input_shapes: Vec<Vec<usize>>,
    output_shapes: Vec<Vec<usize>>,
}

impl OnnxEngine {
    /// Erzeugt eine neue ONNX Engine aus der `Config`.
    /// CPU-Variante – keine CUDA-Abhängigkeiten nötig.
    pub fn new(cfg: &Config, _device_id: Option<usize>) -> Result<Self> {
        // Achtung: ort v2 verlangt globale Init *vor* Session-Erstellung.
        // Mache das einmal ganz früh in deinem Programm (z. B. in main/lib.rs):
        //   ort::init().commit()?;
        //
        // Falls du es hier trotzdem aufrufen willst, ist das idempotent genug,
        // solange vor der ersten Session kein anderes init() mit EPs erfolgte.

        // Builder (v2) – kein Environment-Objekt mehr
        let mut builder = SessionBuilder::new()
            .with_context(|| "Fehler beim Erstellen des SessionBuilder")?;
        builder = builder.with_optimization_level(GraphOptimizationLevel::Level3)?;

        // Modell laden
        let session = builder
            .commit_from_file(&cfg.model.model_path)
            .with_context(|| format!("ONNX-Modell konnte nicht geladen werden: {}", cfg.model.model_path))?;

        // Konsistenz I/O-Listen prüfen
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

    /// Führt eine Inferenz durch.
    /// Erwartet `ArrayD<f32>` mit Shape wie in `model.input_shapes[0]`.
    fn infer_array(&mut self, input: ArrayD<f32>) -> Result<ArrayD<f32>> {
        let mut session = self.session.lock().unwrap();

        // Input gegen Spec validieren
        let expected_in = &self.input_shapes[0];
        anyhow::ensure!(
            input.shape() == expected_in.as_slice(),
            "ONNX: Input-Shape passt nicht. Erwartet {:?}, bekommen {:?}",
            expected_in, input.shape()
        );

        // Tensor aus ndarray erstellen (besitzt die Daten)
        let input_tensor: Tensor<f32> = Tensor::from_array(input.into_owned())?;

        // Session ausführen – Inputs mit Namen übergeben
        let outputs = session.run(ort::inputs![
            &*self.input_names[0] => input_tensor
        ])?;

        // Erstes Output holen (per Name oder Index) und in Array konvertieren
        let dyn_out: &DynValue = &outputs[0]; // oder: &outputs[&*self.output_names[0]]
        let out_view = dyn_out
            .try_extract_array()
            .map_err(|_| anyhow::anyhow!("ONNX: Output ist kein Tensor<f32>"))?;

        // Output-Shape prüfen & in owned Array kopieren
        let expected_out = &self.output_shapes[0];
        anyhow::ensure!(
            out_view.shape() == expected_out.as_slice(),
            "ONNX: Output-Shape passt nicht. Erwartet {:?}, bekommen {:?}",
            expected_out, out_view.shape()
        );

        Ok(out_view.to_owned())
    }
}
