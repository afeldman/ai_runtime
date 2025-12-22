# OmniEngine Documentation

<p align="center">
  <img src="logo.png" alt="OmniEngine Logo" width="200"/>
</p>

## Overview

OmniEngine is a modular AI inference runtime supporting multiple backends (ONNX, TensorRT, TorchScript, TensorFlow) with dynamic batching, Python scripting support, and Redis integration.

## Documentation Index

- [API Documentation](../target/doc/omniengine/index.html) - Generated rustdoc
- [Configuration Guide](config.md) - Runtime configuration
- [Architecture](architecture.md) - System design and components
- [Examples](examples.md) - Usage examples and tutorials

## Quick Links

- [Main README](../README.md)
- [Cargo.toml](../Cargo.toml)
- [Source Code](../src/)

## Building Documentation

Generate the API documentation with:

```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo doc --no-deps --open
```

## Contributing

See the main [README](../README.md) for development setup and contribution guidelines.
