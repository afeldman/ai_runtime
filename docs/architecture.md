# Architecture

<p align="center">
  <img src="logo.png" alt="OmniEngine Architecture" width="400"/>
</p>

## System Overview

OmniEngine is a high-performance inference runtime designed for production ML deployments with multiple backend support and automatic batching.

## Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                         │
│  (HTTP API / gRPC / Redis Queue / Python Bindings)          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      Job Dispatcher                         │
│  • Round-robin distribution                                 │
│  • Load balancing across workers                            │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │Worker 0 │    │Worker 1 │    │Worker N │
    │ GPU 0   │    │ GPU 1   │    │ CPU     │
    └────┬────┘    └────┬────┘    └────┬────┘
         │              │              │
         ▼              ▼              ▼
    ┌─────────────────────────────────────┐
    │         Dynamic Batcher             │
    │  • Collect jobs (max_batch)         │
    │  • Timeout-based flush              │
    │  • Automatic padding                │
    └────────────────┬────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────────┐
    │          Pipeline                   │
    │  ┌──────────────────────────────┐   │
    │  │  Preprocessing               │   │
    │  │  (Python or Rust)            │   │
    │  └──────────────┬───────────────┘   │
    │                 ▼                   │
    │  ┌──────────────────────────────┐   │
    │  │  Inference Engine            │   │
    │  │  (ONNX/TRT/Torch/TF)         │   │
    │  └──────────────┬───────────────┘   │
    │                 ▼                   │
    │  ┌──────────────────────────────┐   │
    │  │  Postprocessing              │   │
    │  │  (Python or Rust)            │   │
    │  └──────────────┬───────────────┘   │
    └─────────────────┼───────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────┐
    │         Redis Storage               │
    │  • JSON serialization               │
    │  • Timestamped results              │
    │  • Key prefix: results:{job_id}     │
    └─────────────────────────────────────┘
```

## Core Components

### 1. Job Dispatcher (`lib.rs`)

- Receives jobs from input channels
- Distributes to workers using round-robin
- Maintains worker queues

### 2. Workers (`worker.rs`)

- One worker per device (GPU/CPU)
- Owns inference engine instance
- Processes batches sequentially

### 3. Dynamic Batcher (`batcher.rs`)

- Collects jobs up to `max_batch`
- Flushes on timeout (`max_wait_ms`)
- Pads batches to required size
- Filters dummy outputs before storage

### 4. Pipeline (`pipeline.rs`)

- Pre/post-processing abstraction
- Trait-based extensibility
- Python integration via PyO3

### 5. Engines (`engine/*.rs`)

- Backend abstraction via `Engine` trait
- Factory pattern for runtime selection
- Device management (CPU/GPU)

### 6. Storage (`storage/redis_store.rs`)

- Async Redis client
- JSON serialization
- Bulk write operations

## Data Flow

1. **Job Submission**: Client sends job to dispatcher
2. **Distribution**: Dispatcher assigns to worker queue
3. **Batching**: Worker collects jobs into batch
4. **Preprocessing**: Pipeline transforms input
5. **Inference**: Engine executes model
6. **Postprocessing**: Pipeline transforms output
7. **Storage**: Results written to Redis
8. **Client Retrieval**: Client polls Redis for results

## Scalability

- **Horizontal**: Multiple runtime instances with shared Redis
- **Vertical**: Multi-GPU via worker pool
- **Async**: Tokio-based concurrency
- **Batching**: Automatic throughput optimization

## Extension Points

- Custom preprocessors/postprocessors (Python or Rust)
- New backend engines (implement `Engine` trait)
- Alternative storage backends (implement storage interface)
- Custom job routing strategies
