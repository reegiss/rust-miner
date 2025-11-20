# Modular Mining Algorithms Architecture

## Overview

The `rust-miner` project implements a **modular, trait-based architecture** for mining algorithms. This allows seamless integration of new algorithms while maintaining code quality and performance.

## Supported Algorithms

### 1. QHash (Qubitcoin Proof-of-Work)
- **Status**: ✅ Fully implemented on CUDA
- **Description**: Quantum-inspired PoW using analytical circuit simulation
- **Usage**: `--algo qhash`
- **Performance**: ~295 MH/s on GTX 1660 SUPER
- **Reference**: [QubitCoin](https://github.com/super-quantum/qubitcoin)

### 2. Ethash (Ethereum Classic)
- **Status**: 🚧 Placeholder implementation (CPU-based)
- **Description**: Ethereum's Proof-of-Work algorithm
- **Usage**: `--algo ethash`
- **Performance**: TBD (currently uses CPU fallback)
- **TODO**: Full CUDA implementation with DAG memory management
- **Reference**: [Ethereum Ethash](https://github.com/ethereum/wiki/wiki/Ethash)

## Architecture

### Trait-Based Design

All mining algorithms implement the `MiningBackend` trait:

```rust
pub trait MiningBackend: Send + Sync {
    fn initialize(&mut self) -> Result<()>;
    
    fn mine_job(
        &self,
        job: &StratumJob,
        extranonce1: &str,
        extranonce2: &[u8],
        start_nonce: u32,
        num_nonces: u32,
    ) -> Result<MiningResult>;
    
    fn device_info(&self) -> Result<GpuInfo>;
    fn last_kernel_ms(&self) -> u64;
}
```

### Module Structure

```
src/
├── algorithms/
│   ├── mod.rs              # Algorithm trait definition
│   ├── qhash.rs            # QHash CPU implementation (reference)
│   └── ethash.rs           # Ethash CPU implementation (reference)
├── cuda/
│   ├── mod.rs              # CUDA utilities and device management
│   ├── qhash.cu            # QHash CUDA kernel
│   ├── qhash_backend.rs    # QHash backend implementing MiningBackend
│   ├── ethash.cu           # Ethash CUDA kernel (placeholder)
│   └── ethash_backend.rs   # Ethash backend implementing MiningBackend
└── main.rs
    ├── create_backend_for_device_sync() # Algorithm dispatcher
    └── gpu_mining_task()               # Mining orchestration
```

## Adding a New Algorithm

### Step 1: Create Algorithm Reference Implementation

Create `src/algorithms/newalgo.rs`:

```rust
use super::_HashAlgorithm;

pub struct NewAlgo;

impl NewAlgo {
    pub fn new() -> Self {
        Self
    }
}

impl _HashAlgorithm for NewAlgo {
    fn name(&self) -> &str {
        "newalgo"
    }

    fn hash(&self, header: &[u8; 80]) -> [u8; 32] {
        // Implement your hashing algorithm here
        // Must accept 80-byte block header, return 32-byte hash
        todo!()
    }

    fn meets_target(&self, hash: &[u8; 32], target: &[u8; 32]) -> bool {
        // Compare hash against target (big-endian)
        todo!()
    }
}
```

Register in `src/algorithms/mod.rs`:

```rust
pub mod newalgo;
```

### Step 2: Create CUDA Kernel (if GPU acceleration needed)

Create `src/cuda/newalgo.cu`:

```cuda
extern "C" __global__ void newalgo_mine(
    const unsigned char* block_header,
    unsigned int nonce_start,
    unsigned int* solution,
    unsigned int difficulty_target
) {
    // CUDA kernel implementation
    // Each thread processes one nonce
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int nonce = nonce_start + gid;
    
    // Compute hash for this nonce
    // If hash meets difficulty, store nonce in solution
}
```

### Step 3: Create Backend Implementation

Create `src/cuda/newalgo_backend.rs`:

```rust
use anyhow::Result;
use crate::backend::{MiningBackend, MiningResult, GpuInfo};
use crate::stratum::StratumJob;
use crate::mining::{calculate_merkle_root, nbits_to_target, hex_to_u32_le, hex_to_bytes_be};

pub struct NewalgoCudaBackend {
    // GPU-specific state
}

impl NewalgoCudaBackend {
    pub fn new(device_index: usize) -> Result<Self> {
        // Initialize GPU resources
        todo!()
    }
}

impl MiningBackend for NewalgoCudaBackend {
    fn initialize(&mut self) -> Result<()> {
        // Initialize GPU if needed
        Ok(())
    }

    fn mine_job(
        &self,
        job: &StratumJob,
        extranonce1: &str,
        extranonce2: &[u8],
        start_nonce: u32,
        num_nonces: u32,
    ) -> Result<MiningResult> {
        // Build 80-byte block header
        let mut header = [0u8; 80];
        
        // ... populate header fields ...
        
        // Mine the job
        let found_nonce: Option<u32> = todo!("Call GPU kernel");
        
        Ok(MiningResult {
            found_share: found_nonce.is_some(),
            nonce: found_nonce,
            hash: todo!(),
            hashes_computed: num_nonces as u64,
        })
    }

    fn device_info(&self) -> Result<GpuInfo> {
        // Return GPU information
        todo!()
    }

    fn last_kernel_ms(&self) -> u64 {
        // Return last kernel execution time
        0
    }
}
```

Register in `src/cuda/mod.rs`:

```rust
mod newalgo_backend;
pub use newalgo_backend::NewalgoCudaBackend;
```

### Step 4: Add to Backend Dispatcher

Update `src/main.rs::create_backend_for_device_sync()`:

```rust
fn create_backend_for_device_sync(algo: &str, device_index: usize) -> Result<...> {
    match algo {
        "qhash" => { /* ... */ },
        "ethash" => { /* ... */ },
        "newalgo" => {
            let mut backend = cuda::NewalgoCudaBackend::new(device_index)?;
            backend.initialize()?;
            let device_info = backend.device_info()?;
            let boxed: Box<dyn MiningBackend> = Box::new(backend);
            Ok((std::sync::Arc::new(tokio::sync::Mutex::new(boxed)), device_info))
        }
        _ => {
            anyhow::bail!("Unsupported algorithm: {}. Supported algorithms: qhash, ethash, newalgo", algo);
        }
    }
}
```

### Step 5: Update CLI Help (if needed)

Edit `src/cli.rs` to update algorithm descriptions if adding special CLI flags.

### Step 6: Add Dependencies (if needed)

Update `Cargo.toml` if your algorithm needs additional cryptographic libraries:

```toml
[dependencies]
# For new algorithm dependencies
new-algo-lib = "1.0"
```

## Example Usage

```bash
# Mine QubitCoin with QHash
./target/release/rust-miner \
    --algo qhash \
    --url pool.example.com:8610 \
    --user wallet.worker \
    --pass x

# Mine Ethereum Classic with Ethash (simplified)
./target/release/rust-miner \
    --algo ethash \
    --url ethermine.org:4444 \
    --user wallet.worker \
    --pass x
```

## Performance Optimization Tips

### CUDA Optimization

1. **Occupancy**: Tune `threads_per_block` to maximize SM utilization
2. **Shared Memory**: Use `__shared__` for frequently accessed data (block header)
3. **Unrolling**: Use `#pragma unroll` in hot loops
4. **Coalescing**: Access GPU memory in coalesced patterns
5. **Divergence**: Minimize branch divergence in kernels

### Kernel Launch Strategy

```cuda
// Recommended config for mining kernels
const int THREADS_PER_BLOCK = 256;
const int NUM_BLOCKS = (nonces + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

LaunchConfig {
    grid_dim: (NUM_BLOCKS, 1, 1),
    block_dim: (THREADS_PER_BLOCK, 1, 1),
    shared_mem_bytes: 0,  // Adjust based on algorithm
}
```

### Memory Transfer Optimization

- Batch multiple nonce ranges in a single kernel launch
- Use pinned memory for CPU-GPU transfers
- Minimize synchronous transfers (prefer async when possible)

## Testing New Algorithms

```bash
# Build with new algorithm
cargo build --release

# Run in benchmark mode (no pool needed)
./target/release/rust-miner \
    --algo newalgo \
    --benchmark

# Enable debug logging
RUST_LOG=debug ./target/release/rust-miner \
    --algo newalgo \
    --url pool.example.com:3333 \
    --user test
```

## Future Algorithms

Potential algorithms for implementation:

1. **KawPoW** - Ravencoin's PoW
2. **Autolykos** - Ergo blockchain PoW
3. **RandomHash** - PASCAL blockchain PoW
4. **CycleHash** - Grin's Cuckoo Cycle variant
5. **ProgPow** - Programmable PoW

Each follows the same modular pattern - implement the trait, add a CUDA kernel, and register with the dispatcher.

---

**Status**: November 20, 2025
**Author**: rust-miner team
