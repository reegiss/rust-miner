# Copilot Instructions for rust-miner (CUDA-only)

## Project Overview
This is a high-performance Rust-based cryptocurrency mining application with GPU acceleration. The project is CUDA-only and targets NVIDIA GPUs.

**Cross-Platform Support**: Linux and Windows are both first-class targets. All code must be platform-agnostic or use conditional compilation when platform-specific features are required.

**GPU Required**: This application requires an NVIDIA GPU with CUDA support for mining. There is no CPU or OpenCL fallback. Systems without a compatible CUDA GPU cannot mine.

## Development Setup

### Building and Running
```bash
# Build the project (cross-platform)
cargo build

# Run in development mode
cargo run

# Run with release optimizations (CPU only)
cargo build --release && ./target/release/rust-miner  # Linux/macOS
cargo build --release && .\target\release\rust-miner.exe  # Windows

# Build with CUDA support (NVIDIA GPUs)
cargo build --release --features cuda

# Run tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run CUDA-specific tests
cargo test --features cuda

# Check code without building
cargo check

# Format code
cargo fmt

# Run linter
cargo clippy -- -D warnings

# Benchmark CUDA performance
cargo bench --bench mining_bench --features cuda
```

### Platform-Specific Considerations

#### Linux
- Preferred development platform
- Native CUDA support via CUDA Toolkit
- Use `perf` for profiling
- Shared library extension: `.so`

#### Windows
- CUDA via CUDA Toolkit for Windows
- Use Windows Performance Analyzer for profiling
- Shared library extension: `.dll`
- Consider using WSL2 for Linux-like development experience (CUDA on WSL supported with NVIDIA drivers)

#### Cross-Platform Best Practices
- **Path Handling**: Always use `std::path::Path` and `PathBuf`, never hardcode separators
```rust
// Good - cross-platform
use std::path::PathBuf;
let config_path = PathBuf::from("config").join("settings.toml");

// Bad - Linux-only
let config_path = "config/settings.toml";
```

- **Line Endings**: Let `.gitattributes` handle it, use `\n` in code
- **File System**: Use `std::fs` and handle case sensitivity differences
- **Process Spawning**: Use `std::process::Command` with platform-specific args
```rust
#[cfg(target_os = "windows")]
const NVCC: &str = "nvcc.exe";

#[cfg(not(target_os = "windows"))]
const NVCC: &str = "nvcc";
```

- **Dynamic Libraries**: Use conditional compilation for library loading
```rust
#[cfg(target_os = "linux")]
const CUDA_LIB: &str = "libcudart.so";

#[cfg(target_os = "windows")]
const CUDA_LIB: &str = "cudart64_XX.dll";
```

## Code Conventions

### Rust-Specific Patterns
- **Error Handling**: Use `Result<T, E>` for recoverable errors and `panic!` only for unrecoverable states
  - Create custom error types with `thiserror` for better error context
  - Use `anyhow` for application-level error handling
  - Propagate errors with `?` operator instead of `unwrap()`
- **Ownership**: Follow Rust ownership rules strictly; prefer borrowing over cloning when possible
  - Use `&` for read-only access, `&mut` for exclusive modification
  - Clone only when necessary (e.g., moving data across threads)
  - Leverage `Cow<'a, T>` for flexible ownership when data may or may not be owned
- **Module Structure**: Organize code into logical modules (e.g., `mining`, `blockchain`, `network`, `crypto`)
- **Async Code**: If using async, prefer `tokio` runtime with `async/await` syntax
  - Use `tokio::spawn` for concurrent tasks
  - Prefer bounded channels (`mpsc::channel(size)`) to prevent unbounded memory growth
  - Use `tokio::select!` for cancellation-aware operations

### High-Performance Best Practices

#### Memory Management
- **Zero-Copy Operations**: Use `&[u8]` slices instead of `Vec<u8>` when possible
- **Stack Allocation**: Prefer stack-allocated arrays for small, fixed-size data
- **Arena Allocation**: Consider `bumpalo` or `typed-arena` for bulk allocations
- **Avoid Allocations in Hot Paths**: Pre-allocate buffers and reuse them
```rust
// Good: Reuse buffer
let mut buffer = vec![0u8; 1024];
loop {
    hash_data(&mut buffer);
}

// Bad: Allocate every iteration
loop {
    let buffer = vec![0u8; 1024];
}
```

#### CPU Optimization
- **SIMD**: Use `packed_simd` or `std::arch` for vectorized operations on hashes
- **Branch Prediction**: Minimize branches in hot loops; use lookup tables when possible
- **Cache Locality**: Keep frequently accessed data together in memory
- **Inline Critical Functions**: Use `#[inline]` or `#[inline(always)]` for small, hot functions
```rust
#[inline(always)]
fn compute_hash_round(state: &mut [u32; 8], data: &[u8]) {
    // Critical hashing logic
}
```

#### Parallelism Strategy
- **Data Parallelism**: Use `rayon` for parallel iterators on independent work
```rust
use rayon::prelude::*;
candidates.par_iter()
    .find_map_any(|nonce| try_mine(nonce))
```
- **Task Parallelism**: Use `tokio` for I/O-bound operations (network, disk)
- **Thread Pools**: Create dedicated thread pools for mining vs. network operations
- **Lock-Free Structures**: Use `crossbeam` or `parking_lot` for concurrent data structures
  - `parking_lot::RwLock` is faster than `std::sync::RwLock`
  - Use `AtomicUsize` / `AtomicU64` for simple counters

#### Data Structures
- **Choose Wisely**:
  - `Vec<T>`: Sequential access, cache-friendly
  - `HashMap<K, V>`: Use `ahash::AHashMap` for better performance than default hasher
  - `BTreeMap<K, V>`: When you need sorted keys
  - `SmallVec`: Stack-allocate small vectors to avoid heap allocation
- **Capacity Pre-allocation**: Always `Vec::with_capacity()` if size is known
- **Avoid Dynamic Dispatch**: Prefer generics over trait objects in hot paths

#### GPU Acceleration Patterns
- **CUDA (NVIDIA)**: Single supported backend for maximum performance
  - `cudarc` crate for CUDA support (modern, safe Rust bindings)
  - Direct access to NVIDIA GPU features and optimizations
  - Best performance on NVIDIA hardware (GTX/RTX series)
- **No CPU/OpenCL Fallback**: CUDA GPU is mandatory
  - Application will exit gracefully if no compatible GPU is detected
  - Display helpful error message directing user to CUDA requirements
- **Hybrid CPU/GPU Strategy**: Balance GPU workload with CPU coordination
```rust
// Example: GPU handles all mining, CPU coordinates
let gpu_available = detect_gpu_support()?;

// GPU does the mining
spawn_gpu_miners(0..u64::MAX)?;

// CPU handles coordination tasks only (not mining)
// - Network communication
// - Result validation
// - Statistics tracking
```

- **Memory Transfer Optimization**: Minimize CPU↔GPU data transfers
  - Batch multiple blocks for GPU processing
  - Keep kernel data on GPU between iterations
  - Use pinned memory for faster transfers
```rust
// Good: Batch processing
let batch_size = 1000;
let blocks_batch: Vec<BlockHeader> = blocks.take(batch_size).collect();
gpu_kernel.process_batch(&blocks_batch)?;

// Bad: Individual transfers
for block in blocks {
    gpu_kernel.process_single(&block)?; // Transfer overhead!
}
```

- **Kernel Optimization**:
  - Maximize occupancy: Balance threads per block vs. register usage
  - Use shared memory for frequently accessed data (block header)
  - Avoid divergent branches in GPU kernels
  - Unroll loops when iteration count is known
```rust
// CUDA kernel structure for mining (preferred)
const CUDA_KERNEL: &str = r#"
extern "C" __global__ void mine_block(
    const unsigned char* block_header,
    unsigned int nonce_start,
    unsigned int* solution,
    unsigned int difficulty_target
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int nonce = nonce_start + gid;
    
    // Copy header to shared memory (faster access)
    __shared__ unsigned char shared_header[80];
    if (threadIdx.x < 80) {
        shared_header[threadIdx.x] = block_header[threadIdx.x];
    }
    __syncthreads();
    
    // Hash computation with nonce
    unsigned char hash[32];
    sha256d(shared_header, nonce, hash);
    
    // Check if solution meets difficulty
    if (meets_difficulty(hash, difficulty_target)) {
        atomicCAS(solution, 0, nonce);
    }
}
"#;

```

- **Multi-GPU Support**: Scale across multiple GPUs
```rust
pub struct MultiGpuMiner {
    devices: Vec<GpuDevice>,
}

impl MultiGpuMiner {
    pub fn distribute_work(&self, nonce_space: Range<u64>) -> Vec<Range<u64>> {
        let total_compute_power: u32 = self.devices.iter()
            .map(|d| d.compute_units)
            .sum();
        
        // Distribute proportionally to compute capability
        let mut ranges = Vec::new();
        let mut current = nonce_space.start;
        
        for device in &self.devices {
            let share = (nonce_space.len() as f64 * device.compute_units as f64 
                        / total_compute_power as f64) as u64;
            ranges.push(current..current + share);
            current += share;
        }
        
        ranges
    }
}
```

- **Backend Strategy**: CUDA-only. If CUDA initialization fails, exit with a clear error message and setup hints.

- **Performance Monitoring**: Track GPU metrics
  - Monitor GPU temperature and throttling
  - Track memory bandwidth utilization
  - Measure kernel execution time vs. transfer time
  - Compare hash rates: GPU vs. CPU

### Architecture Principles

#### Layered Architecture
```
┌─────────────────────────────────────┐
│   CLI / API Interface Layer         │
├─────────────────────────────────────┤
│   Business Logic / Mining Engine    │
├─────────────────────────────────────┤
│   Core Primitives (Hash, Block)     │
├─────────────────────────────────────┤
│   Infrastructure (Network, Storage) │
└─────────────────────────────────────┘
```

#### Dependency Injection
- Use trait-based abstractions for testability
```rust
pub trait BlockchainClient {
    fn submit_block(&self, block: Block) -> Result<()>;
}

pub struct MiningEngine<C: BlockchainClient> {
    client: C,
}
```

#### Separation of Concerns
- **Domain Logic**: Pure functions without I/O
- **Infrastructure**: Networking, persistence, external APIs
- **Configuration**: Centralized in `Config` struct, loaded once
- **State Management**: Minimize shared mutable state; prefer message-passing

## Architecture Guidelines

### Expected Components
When developing this project, consider these typical mining application components:

- **Mining Engine** (`src/mining/`): GPU-only hashing/proof-of-work logic
  - `engine.rs`: Main mining loop and work distribution
  - `cuda.rs`: CUDA backend implementation (only)
  - `backend.rs`: Backend trait and auto-detection
- **Blockchain Interface** (`src/blockchain/`): Connection to blockchain network
  - `client.rs`: Network communication with blockchain nodes
  - `block.rs`: Block structure and validation
  - `transaction.rs`: Transaction handling
- **Configuration** (`src/config.rs`): Runtime configuration management
  - Load from TOML/JSON files
  - Environment variable overrides
  - Validation on startup
  - GPU detection and selection
- **Logging** (`src/logging.rs`): Structured logging for monitoring performance
  - Use `tracing` for async-aware logging
  - Include metrics: hash rate, blocks found, network latency, GPU temp
- **CLI** (`src/cli.rs`): Command-line interface for user interaction
  - Use `clap` with derive macros for argument parsing
  - GPU listing and selection
  - Mining pool configuration

### Module Organization
```
src/
├── main.rs              # Entry point, setup runtime
├── lib.rs               # Library interface for testing
├── config.rs            # Configuration types
├── error.rs             # Custom error types
├── mining/
│   ├── mod.rs           # Public API
│   ├── engine.rs        # Mining logic
│   ├── backend.rs       # Backend trait and detection
│   └── cuda.rs          # CUDA implementation (only backend)
├── blockchain/
│   ├── mod.rs
│   ├── client.rs        # Network client
│   ├── types.rs         # Block, Transaction types
│   └── validation.rs    # Validation logic
├── network/
│   ├── mod.rs
│   ├── protocol.rs      # Protocol implementation
│   └── peers.rs         # Peer management
└── utils/
    ├── mod.rs
    ├── metrics.rs       # Performance metrics
    └── gpu.rs           # GPU detection and info
```

### Performance Profiling Workflow
```bash
# Profile with perf (Linux)
cargo build --release
perf record --call-graph=dwarf ./target/release/rust-miner
perf report

# Profile with flamegraph
cargo install flamegraph
cargo flamegraph

# Benchmark critical paths
cargo bench --bench mining_bench

# Memory profiling with valgrind
valgrind --tool=massif ./target/release/rust-miner
```

### Recommended Dependencies
```toml
[dependencies]
# Error handling
thiserror = "1.0"
anyhow = "1.0"

# Async runtime
tokio = { version = "1.35", features = ["full"] }

# Parallelism
rayon = "1.8"
crossbeam = "0.8"
num_cpus = "1.16"

# Performance
parking_lot = "0.12"      # Faster locks
ahash = "0.8"             # Faster hashing
smallvec = "1.11"         # Stack-allocated vecs

# GPU Acceleration (enabled via features)
cudarc = { version = "0.11", optional = true }        # CUDA support (NVIDIA)
cuda-sys = { version = "0.3", optional = true }       # Low-level CUDA bindings

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Logging
tracing = "0.1"
tracing-subscriber = "0.3"

# CLI
clap = { version = "4.4", features = ["derive"] }

# Crypto
sha2 = "0.10"
blake3 = "1.5"            # Fast hashing for mining

# Cross-platform utilities
dirs = "5.0"              # Standard user directories (cross-platform)
which = "6.0"             # Find executables in PATH (cross-platform)

[features]
default = ["cuda"]                                     # Default to CUDA (NVIDIA GPUs)
cuda = ["dep:cudarc", "dep:cuda-sys"]                 # CUDA backend (only)


[target.'cfg(windows)'.dependencies]
# Windows-specific dependencies
windows-sys = { version = "0.52", features = ["Win32_System_Threading", "Win32_Foundation"] }

[target.'cfg(unix)'.dependencies]
# Unix/Linux-specific dependencies (if needed)
libc = "0.2"
```

### Testing Strategy
- Unit tests: Place in the same file as the code using `#[cfg(test)]` modules
- Integration tests: Place in `tests/` directory
- Test critical paths: hashing functions, network communication, data validation
- **Platform-specific tests**: Use `#[cfg(target_os = "...")]` for OS-specific functionality
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_cross_platform_feature() {
        // Test that works on all platforms
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_linux_specific() {
        // Linux-only test
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn test_windows_specific() {
        // Windows-only test
    }
}
```

## Dependencies Management
- Keep `Cargo.toml` organized with comments for dependency groups
- Pin versions for critical dependencies
- Document why non-standard dependencies are used

## Common Tasks

### Adding a New Feature Module
1. Create new file in `src/` (e.g., `src/mining.rs`)
2. Declare module in `src/main.rs` or `src/lib.rs`: `mod mining;`
3. Define public API with `pub` keyword
4. Add tests in `#[cfg(test)]` module
5. Update this file with new patterns/conventions

### Debugging
```bash
# Run with debug logging
RUST_LOG=debug cargo run

# Run with backtrace on panic
RUST_BACKTRACE=1 cargo run

# Use rust-gdb or rust-lldb for debugging
rust-gdb ./target/debug/rust-miner
```

## Project-Specific Notes
As this codebase grows, document:
- Specific cryptographic algorithms used
- Network protocol details
- Configuration file formats
- Performance benchmarks and optimization strategies
- Integration with specific blockchain networks

---
*This file should be updated as the project evolves. Keep it concise and focused on non-obvious patterns.*
