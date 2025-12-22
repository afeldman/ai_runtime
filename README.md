# OmniEngine

<p align="center">
  <img src="docs/logo.png" alt="OmniEngine Logo" width="200"/>
</p>

<p align="center">
  ⚡ A modular AI inference runtime for ONNX, TensorRT, TorchScript, and TensorFlow<br/>
  with plugin-based pre/post-processing and Python scripting support.
</p>

---

## Features

- **Multi-backend Inference**

  - [ONNX Runtime](https://onnxruntime.ai/) (CPU, CUDA)
  - [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) (GPU, optimized)
  - [TorchScript](https://pytorch.org/docs/stable/jit.html) (CPU/GPU)
  - [TensorFlow](https://www.tensorflow.org/) (CPU/GPU)

- **Batching**

  - Dynamic job batching via Tokio channels
  - Automatic padding to fixed batch sizes

- **Pipeline Plugins**

  - Pre- and post-processing as composable plugins
  - Built-in: `Normalize`, `Resize`, `Softmax`, `Argmax`
  - Python scripting support for custom plugins
  - Identity fallback (no-op)

- **Redis Integration**

  - Job input/output exchange via Redis
  - JSON serialization of results
  - Timestamped results for downstream systems

- **Python Bindings**
  - Exposed via [PyO3](https://pyo3.rs/)
  - Installable with [maturin](https://github.com/PyO3/maturin)

---

## Architecture

```markdown
      ┌───────────┐
      │  Clients  │
      └─────┬─────┘
            │ Jobs (Redis Queue)
            ▼
     ┌───────────────────┐
     │      Dispatcher   │
     └──────┬────────────┘
            │ Round-robin

┌───────────┼─────────────┐
▼           ▼             ▼
Worker 0 Worker 1 ... Worker N
(GPU/CPU) (GPU/CPU) (GPU/CPU)
└── Preprocess → Engine → Postprocess → Redis Output
```

## Quickstart

### 1. Install dependencies

Make sure you have Rust and Python set up:

```bash
# Rust toolchain
rustup default stable

# Python virtual environment
python -m venv .venv
source .venv/bin/activate
pip install maturin
```

### 2. Build the library

```bash
maturin develop --release
```

### 3. Run the runtime

Create a runtime.toml:

```toml
[model]
backend = "onnx"
device = "cpu"
model_path = "model.onnx"
input_names = ["input"]
input_shapes = [[1, 1, 28, 28]]
output_names = ["output"]
output_shapes = [[1, 10]]

[queue]
max_batch = 8
max_wait_ms = 20

[redis]
url = "redis://127.0.0.1/"
out_prefix = "results:"
```

#### Start the engine:

```bash
cargo run --release
```

#### Python Usage

```python
import omniengine
import numpy as np

# Direct inference using Python bindings
engine = omniengine.PyOnnxEngine("runtime.toml")
x = np.random.randn(1, 3, 224, 224).astype(np.float32)
output = engine.infer(x)
print(f"Output shape: {output.shape}")

# Alternatively, send jobs via Redis queue
import redis
import cbor2
import uuid

r = redis.Redis(host='localhost', port=6379, db=0)

# Prepare job
job_id = str(uuid.uuid4())
job = {
    "id": job_id,
    "input": x.tobytes(),
    "shape": list(x.shape),
    "dtype": "f32"
}

# Submit to queue and wait for result
r.lpush("inference_queue", cbor2.dumps(job))
result = r.blpop(f"results:{job_id}", timeout=10)

if result:
    output_data = cbor2.loads(result[1])
    print(f"Result: {output_data}")
```

## Development

- Run tests:
  ```bash
  cargo test
  ```
- Format code:
  ```bash
  cargo fmt
  ```
- Check lints:
  ```bash
  cargo clippy
  ```

## Documentation

Comprehensive documentation is available in the [docs/](docs/) directory:

- **[Configuration Guide](docs/config.md)** - Runtime configuration and backend setup
- **[Architecture](docs/architecture.md)** - System design and components
- **[Examples](docs/examples.md)** - Usage examples and tutorials

### API Documentation

Build the API docs with rustdoc:

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo doc --no-deps --open
```

Module-level docs:

- [src/lib.rs](src/lib.rs): overview and `start_runtime()`
- [src/engine/](src/engine): backends (`onnx`, `tensorrt`, `torch`, `tensorflow`)
- [src/pipeline.rs](src/pipeline.rs): pre/post-processing traits and `Pipeline`
- [src/batcher.rs](src/batcher.rs): dynamic batching
- [src/storage/redis_store.rs](src/storage/redis_store.rs): Redis integration
- [src/scripting/plugins.rs](src/scripting/plugins.rs): Python-backed processors (PyO3)

## Roadmap

- Add gRPC interface
- Expand TensorFlow backend
- Advanced batching strategies
- More plugin examples (Python + Rust)
- Kubernetes deployment templates

## License

MIT License © 2025 Anton Feldmann
