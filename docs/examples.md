# Examples

## Basic Usage

### 1. Running the Runtime

```bash
# Create runtime.toml
cat > runtime.toml << EOF
[model]
backend = "onnx"
device = "cpu"
model_path = "model.onnx"
input_names = ["input"]
input_shapes = [[4, 3, 224, 224]]
output_names = ["output"]
output_shapes = [[4, 1000]]

[input]
batch = 4
channels = 3
height = 224
width = 224
dtype = "f32"

[queue]
max_batch = 4
max_wait_ms = 100

[redis]
url = "redis://127.0.0.1/"
out_prefix = "results:"
EOF

# Start Redis
docker run -d -p 6379:6379 redis:latest

# Run OmniEngine
cargo run --release
```

### 2. Python Bindings

```python
import omniengine
import numpy as np

# Load engine with TOML config
engine = omniengine.PyOnnxEngine("runtime.toml")

# Create dummy input
x = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run inference
output = engine.infer(x)
print(f"Output shape: {output.shape}")
```

### 3. Custom Python Preprocessor

Create `my_plugins.py`:

```python
import numpy as np

def normalize(x):
    """Normalize to [0, 1] range."""
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    return (x - mean) / std

def softmax(x):
    """Apply softmax to output."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

Use in Rust:

```rust
use omniengine::scripting::plugins::{PythonPreprocessor, PythonPostprocessor};
use omniengine::pipeline::Pipeline;

// Load Python processors
let pre = PythonPreprocessor::new("my_plugins", "normalize")?;
let post = PythonPostprocessor::new("my_plugins", "softmax")?;

// Create pipeline
let pipeline = Pipeline::new(Some(pre), Some(post));
```

## Multi-GPU Setup

```toml
[model]
backend = "onnx"
device = "gpu"
gpu_ids = [0, 1, 2, 3]  # Use 4 GPUs
# ... rest of config
```

The runtime will spawn one worker per GPU automatically.

## Advanced Configuration

### TensorRT with FP16

```toml
[model]
backend = "tensorrt"
device = "gpu"
model_path = "model_fp16.trt"
gpu_ids = [0]
# ... rest of config
```

### TorchScript on CUDA

```toml
[model]
backend = "torch"
device = "gpu"
model_path = "model.pt"
gpu_ids = [0]
# ... rest of config
```

## Testing

```bash
# Run all tests
cargo test --all

# Run specific test
cargo test test_collect_batch

# Run with logging
RUST_LOG=debug cargo test
```

## Benchmarking

```bash
# Build optimized binary
cargo build --release

# Profile with perf (Linux)
perf record -g target/release/omniengine-cli
perf report

# Memory profiling with valgrind
valgrind --tool=massif target/release/omniengine-cli
```

## Docker Deployment

```dockerfile
FROM rust:1.75 as builder

WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/omniengine-cli /usr/local/bin/
COPY runtime.toml /etc/omniengine/runtime.toml

CMD ["omniengine-cli"]
```

Build and run:

```bash
docker build -t omniengine .
docker run -v $(pwd)/runtime.toml:/etc/omniengine/runtime.toml omniengine
```
