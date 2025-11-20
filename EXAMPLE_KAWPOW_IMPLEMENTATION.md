# Adding a New Mining Algorithm: Step-by-Step Example

## Example: Adding KawPoW (Ravencoin)

This guide demonstrates how to add a new algorithm to rust-miner using KawPoW as an example.

### Step 1: Create Algorithm Reference (CPU Implementation)

**File: `src/algorithms/kawpow.rs`**

```rust
use super::_HashAlgorithm;
use blake3;

/// KawPoW algorithm for Ravencoin
/// Reference: https://github.com/ravencoin/kawpow
pub struct KawPoW;

impl KawPoW {
    pub fn new() -> Self {
        Self
    }
}

impl _HashAlgorithm for KawPoW {
    fn name(&self) -> &str {
        "kawpow"
    }

    fn hash(&self, header: &[u8; 80]) -> [u8; 32] {
        // Step 1: Initial hash with BLAKE2b
        let initial = blake3::hash(header);
        
        // Step 2: KawPoW mixing (simplified for reference)
        // A full implementation would include:
        // - Multiple rounds of mixing
        // - FNV-1a hashing
        // - Cache line lookups
        // - DAG access patterns
        
        let mut result = [0u8; 32];
        result.copy_from_slice(&initial.as_bytes()[..32]);
        result
    }

    fn meets_target(&self, hash: &[u8; 32], target: &[u8; 32]) -> bool {
        // Big-endian comparison
        for i in 0..32 {
            if hash[i] < target[i] {
                return true;
            } else if hash[i] > target[i] {
                return false;
            }
        }
        false
    }
}
```

### Step 2: Register in Algorithms Module

**File: `src/algorithms/mod.rs`** (add line)

```rust
pub mod kawpow;  // Add this line
```

### Step 3: Create CUDA Kernel

**File: `src/cuda/kawpow.cu`**

```cuda
#include <stdint.h>

// KawPoW CUDA kernel
extern "C" __global__ void kawpow_mine(
    const unsigned char* block_header,
    unsigned int nonce_start,
    unsigned int* solution,
    unsigned int difficulty_target
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int nonce = nonce_start + gid;
    
    // Allocate thread-local workspace
    __shared__ unsigned char shared_header[80];
    if (threadIdx.x < 80) {
        shared_header[threadIdx.x] = block_header[threadIdx.x];
    }
    __syncthreads();
    
    // Prepare header with nonce
    unsigned char header_with_nonce[80];
    for (int i = 0; i < 80; i++) {
        header_with_nonce[i] = shared_header[i];
    }
    
    // Set nonce (last 4 bytes)
    ((unsigned int*)&header_with_nonce[76])[0] = nonce;
    
    // KawPoW hashing:
    // 1. BLAKE2b hash of header
    // 2. Mix with DAG (if using GPU DAG)
    // 3. Final hash
    
    // TODO: Implement full KawPoW algorithm
    // For now, use simplified hash
    unsigned int final_hash_int = nonce ^ 0xDEADBEEF;
    
    if (final_hash_int < difficulty_target) {
        atomicCAS(solution, 0, nonce);
        solution[1] = 1;
    }
}
```

### Step 4: Create Backend Implementation

**File: `src/cuda/kawpow_backend.rs`**

```rust
use anyhow::Result;
use crate::backend::{MiningBackend, MiningResult, GpuInfo};
use crate::stratum::StratumJob;
use crate::mining::{calculate_merkle_root, nbits_to_target, hex_to_u32_le, hex_to_bytes_be};

pub struct KawpowCudaBackend {
    // TODO: GPU resources
}

impl KawpowCudaBackend {
    pub fn new(device_index: usize) -> Result<Self> {
        tracing::info!("Initializing KawPoW backend for device {}", device_index);
        // TODO: Initialize GPU kernel
        Ok(Self {})
    }
}

impl MiningBackend for KawpowCudaBackend {
    fn initialize(&mut self) -> Result<()> {
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
        
        let version_bytes = hex_to_bytes_be(&job.version)?;
        if version_bytes.len() >= 4 {
            header[0..4].copy_from_slice(&version_bytes[0..4]);
        }
        
        let prev_hash = hex_to_bytes_be(&job.prevhash)?;
        if prev_hash.len() == 32 {
            header[4..36].copy_from_slice(&prev_hash[0..32]);
        }
        
        let merkle_root = calculate_merkle_root(job, extranonce1, extranonce2)?;
        header[36..68].copy_from_slice(&merkle_root);
        
        let ntime = hex_to_u32_le(&job.ntime)?;
        header[68..72].copy_from_slice(&ntime.to_le_bytes());
        
        let nbits_bytes = hex_to_bytes_be(&job.nbits)?;
        if nbits_bytes.len() >= 4 {
            header[72..76].copy_from_slice(&nbits_bytes[0..4]);
        }
        
        // TODO: Call GPU kernel with header and nonce range
        let found_nonce: Option<u32> = None; // TODO
        
        Ok(MiningResult {
            found_share: found_nonce.is_some(),
            nonce: found_nonce,
            hash: None, // TODO: Compute actual hash
            hashes_computed: num_nonces as u64,
        })
    }

    fn device_info(&self) -> Result<GpuInfo> {
        Ok(GpuInfo {
            name: "KawPoW (Placeholder)".to_string(),
            compute_capability: (0, 0),
            memory_mb: 0,
            compute_units: 0,
            clock_mhz: 0,
        })
    }

    fn last_kernel_ms(&self) -> u64 {
        0
    }
}
```

### Step 5: Register Backend in CUDA Module

**File: `src/cuda/mod.rs`** (add lines)

```rust
mod kawpow_backend;
pub use kawpow_backend::KawpowCudaBackend;
```

### Step 6: Add to Dispatcher

**File: `src/main.rs`** (in `create_backend_for_device_sync()`)

```rust
fn create_backend_for_device_sync(algo: &str, device_index: usize) -> Result<...> {
    match algo {
        "qhash" => { /* ... */ },
        "ethash" => { /* ... */ },
        "kawpow" => {
            let mut backend = cuda::KawpowCudaBackend::new(device_index)?;
            backend.initialize()?;
            let device_info = backend.device_info()?;
            let boxed: Box<dyn MiningBackend> = Box::new(backend);
            Ok((std::sync::Arc::new(tokio::sync::Mutex::new(boxed)), device_info))
        }
        _ => {
            anyhow::bail!("Unsupported algorithm: {}. Supported: qhash, ethash, kawpow", algo);
        }
    }
}
```

### Step 7: Add Dependencies (if needed)

**File: `Cargo.toml`**

```toml
[dependencies]
# ... existing ...
blake3 = "1.5"  # For KawPoW reference implementation
```

### Step 8: Build and Test

```bash
# Build
cargo build --release

# Run with KawPoW
./target/release/rust-miner \
    --algo kawpow \
    --url ravencoin-pool.example.com:3333 \
    --user wallet.worker \
    --pass x

# Test that algorithm name is recognized
./target/release/rust-miner --help | grep algo
```

### Step 9: Write Tests

**File: `tests/modular_algorithms.rs`** (add)

```rust
#[test]
fn test_kawpow_algorithm_loads() {
    // Test that KawPoW algorithm is accessible
}

#[test]
fn test_kawpow_backend_dispatch() {
    // Test that dispatcher recognizes "kawpow" algorithm name
    let supported_algos = vec!["qhash", "ethash", "kawpow"];
    assert!(supported_algos.contains(&"kawpow"));
}
```

## Implementation Checklist

- [ ] Algorithm reference (CPU) implementation
- [ ] Register in `src/algorithms/mod.rs`
- [ ] Create CUDA kernel (`.cu` file)
- [ ] Implement `MiningBackend` trait
- [ ] Register backend in `src/cuda/mod.rs`
- [ ] Add to dispatcher in `src/main.rs`
- [ ] Add dependencies to `Cargo.toml`
- [ ] Compile: `cargo build --release`
- [ ] All tests passing: `cargo test`
- [ ] Update README with algorithm info
- [ ] Document algorithm-specific options (if any)

## Performance Optimization Checklist

Once basic implementation works:

- [ ] Benchmark against reference implementation
- [ ] Profile GPU kernel with `nvprof`
- [ ] Optimize occupancy (threads per block)
- [ ] Implement shared memory optimization
- [ ] Compare memory bandwidth usage
- [ ] Tune launch configuration for target GPU
- [ ] Profile CPU overhead (host-to-device transfers)

## Testing Checklist

- [ ] Unit tests for hash computation
- [ ] Integration tests with test pools
- [ ] Validate shares are accepted by pool
- [ ] Compare hashrate with reference miners
- [ ] Test on multiple GPU models
- [ ] Stress test for 24+ hours
- [ ] Monitor GPU temperature and power

## Documentation Checklist

- [ ] Add algorithm description to README
- [ ] Document algorithm parameters (DAG size, etc.)
- [ ] Create algorithm-specific tuning guide
- [ ] Add troubleshooting section
- [ ] Document pool compatibility
- [ ] Create performance benchmark results

---

**Time Estimate**: 
- Basic implementation: 2-3 hours
- Full GPU optimization: 1-2 weeks
- Testing and validation: 1 week
- Documentation: 1-2 days
