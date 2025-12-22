# Configuration Guide

## Runtime Configuration

OmniEngine is configured via a TOML file (`runtime.toml`) with the following sections:

### Model Configuration

```toml
[model]
backend = "onnx"              # Backend: "onnx", "tensorrt", "torch", "tensorflow"
device = "cpu"                # Device: "cpu" or "gpu"
model_path = "model.onnx"     # Path to model file
gpu_ids = [0, 1]              # GPU IDs for multi-GPU (optional)

# Input/Output specifications
input_names = ["input"]
input_shapes = [[1, 3, 224, 224]]
output_names = ["output"]
output_shapes = [[1, 1000]]
```

### Input Configuration

```toml
[input]
batch = 4              # Target batch size
channels = 3           # Number of channels
height = 224           # Image height
width = 224            # Image width
dtype = "f32"          # Data type: "f32", "u8", etc.
```

### Queue Configuration

```toml
[queue]
max_batch = 4          # Maximum jobs to collect per batch
max_wait_ms = 100      # Maximum wait time for batching (ms)
```

### Redis Configuration

```toml
[redis]
url = "redis://127.0.0.1/"
out_prefix = "results:"
```

## Backend-Specific Notes

### ONNX

- CPU execution by default
- Enable CUDA with `onnx-cuda` feature
- Automatic binary download via `download-binaries` feature

### TensorRT

- Requires CUDA and TensorRT installation
- Enable with `tensorrt` feature
- GPU-only backend

### TorchScript

- Supports CPU and CUDA
- Enable with `torch` feature
- Loads `.pt` files

### TensorFlow

- Supports CPU and GPU
- Enable with `tensorflow` feature
- Loads SavedModel or frozen graphs

## Environment Variables

- `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` - Required for building with Python 3.14+
- `RUST_LOG=info` - Set logging level (trace, debug, info, warn, error)
