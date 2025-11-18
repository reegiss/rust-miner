/// CUDA mining implementation
mod qhash_backend;

pub use qhash_backend::QHashCudaBackend;

use anyhow::{anyhow, Result};
use cudarc::driver::*;
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

// Import raw CUDA API for non-blocking polling
use cudarc::driver::sys::lib;
use cudarc::driver::sys::CUresult;

const CUDA_KERNEL_SRC: &str = include_str!("qhash.cu");

/// Internal CUDA miner implementation (used by backends)
#[derive(Clone)]
pub(crate) struct CudaMiner {
    device: Arc<CudaDevice>,
    func: CudaFunction,
    last_kernel_ms: Arc<AtomicU64>, // duration of last kernel for adaptive polling
}

impl CudaMiner {
    /// Create new CUDA miner
    pub(crate) fn new() -> Result<Self> {
        // Initialize CUDA device
        let device = CudaDevice::new(0)?;
        
        tracing::info!("Compiling CUDA kernel for qhash...");
        
        // Compile CUDA kernel to PTX
        let ptx = compile_ptx(CUDA_KERNEL_SRC).map_err(|e| {
            anyhow!("Failed to compile CUDA kernel: {}", e)
        })?;
        
        tracing::info!("Loading CUDA module...");
        
        // Load module and get function
        device.load_ptx(ptx, "qhash", &["qhash_mine"])?;
        let func = device.get_func("qhash", "qhash_mine")
            .ok_or_else(|| anyhow!("Failed to load qhash_mine function"))?;
        
        tracing::info!("CUDA miner initialized successfully");
        
        Ok(Self {
            device,
            func,
            last_kernel_ms: Arc::new(AtomicU64::new(0)),
        })
    }
    
    /// Mine a job on GPU (blocking, low CPU via driver polling + sleep)
    ///
    /// Returns Option<(nonce, hash)> if a valid share is found
    pub(crate) fn mine_job(
        &self,
        block_header: &[u8; 76],  // Header without nonce (76 bytes)
        ntime: u32,                // Network time
        target: &[u8; 32],         // Target difficulty
        start_nonce: u32,
        num_nonces: u32,
    ) -> Result<Option<(u32, [u8; 32])>> {
        // Allocate device memory
        let d_header = self.device.htod_copy(block_header.as_slice().to_vec())?;
        let d_target = self.device.htod_copy(target.as_slice().to_vec())?;
        
        // Solution buffer (initialized to 0xFFFFFFFF = no solution)
        let h_solution = vec![0xFFFFFFFFu32];
        let d_solution = self.device.htod_copy(h_solution.clone())?;
        
        // Hash output buffer (32 bytes, will be populated if solution found)
        let h_found_hash = vec![0u8; 32];
        let d_found_hash = self.device.htod_copy(h_found_hash.clone())?;
        
        // Launch configuration - optimized for QHash
        // OPTIMIZATION PHASE 1: Reduce warp divergence
        // Quantum simulation only uses 16 threads, other 496 are idle waiting in syncthreads()
        // Reducing to 64 threads/block reduces warp divergence overhead
        // This may reduce occupancy but increases per-thread efficiency
        // Test target: compare 37 MH/s baseline vs. 64-thread variant
        let threads_per_block = 64;
        let num_blocks = (num_nonces + threads_per_block - 1) / threads_per_block;
        
        tracing::debug!(
            "Launching kernel: {} blocks x {} threads = {} nonces",
            num_blocks,
            threads_per_block,
            num_nonces
        );
        
        // Launch kernel with optimized shared memory (minimal usage)
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 76 + 32,  // header + target only (108 bytes)
        };
        
        unsafe {
            self.func.clone()
                .launch(
                    cfg,
                    (
                        &d_header,
                        ntime,
                        start_nonce,
                        &d_target,
                        &d_solution,
                        &d_found_hash,
                        num_nonces,
                    ),
                )
                .map_err(|e| anyhow!("Kernel launch failed: {}", e))?;
        }
        
        // Non-blocking driver polling with adaptive sleep (host thread, not async)
        let stream = *self.device.cu_stream();
        let poll_start = std::time::Instant::now();
        let mut poll_iterations = 0u32;
        tracing::debug!("GPU poll start: nonces={}", num_nonces);
        loop {
            let result = unsafe { lib().cuStreamQuery(stream) };
            match result {
                CUresult::CUDA_SUCCESS => {
                    let elapsed = poll_start.elapsed();
                    let ms = (elapsed.as_secs_f64() * 1000.0) as u64;
                    self.last_kernel_ms.store(ms, Ordering::Relaxed);
                    tracing::debug!("GPU poll done: iters={} elapsed_ms={} batch_nonces={}", poll_iterations, ms, num_nonces);
                    break;
                }
                CUresult::CUDA_ERROR_NOT_READY => {
                    poll_iterations += 1;
                    // Aim ~16 polls per kernel, clamp range
                    let last = self.last_kernel_ms.load(Ordering::Relaxed);
                    let interval_ms = if last == 0 { 8 } else { (last / 16).clamp(2, 40) };
                    std::thread::sleep(std::time::Duration::from_millis(interval_ms));
                }
                _ => return Err(anyhow!("Stream query failed: {:?}", result)),
            }
        }
        
        // Copy result back
        let mut h_solution_result = vec![0u32];
        self.device.dtoh_sync_copy_into(&d_solution, &mut h_solution_result)?;
        
        let found_nonce = h_solution_result[0];
        
        if found_nonce == 0xFFFFFFFF {
            // No solution found
            Ok(None)
        } else {
            // Solution found! Read hash from GPU (no CPU recomputation needed)
            let mut h_found_hash_result = vec![0u8; 32];
            self.device.dtoh_sync_copy_into(&d_found_hash, &mut h_found_hash_result)?;
            
            let hash: [u8; 32] = h_found_hash_result
                .try_into()
                .map_err(|_| anyhow!("Failed to convert hash vector to array"))?;
            
            Ok(Some((found_nonce, hash)))
        }
    }
    
    /// Get device name
    pub(crate) fn device_name(&self) -> Result<String> {
        Ok(self.device.name()?)
    }
    
    /// Get device compute capability
    pub(crate) fn compute_capability(&self) -> Result<(i32, i32)> {
        use cudarc::driver::sys::CUdevice_attribute;
        let major = self.device.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor = self.device.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        Ok((major, minor))
    }

    /// Get last kernel duration in milliseconds (0 if none yet)
    pub(crate) fn last_kernel_ms(&self) -> u64 {
        self.last_kernel_ms.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_cuda_miner_init() {
        let miner = CudaMiner::new();
        assert!(miner.is_ok(), "Failed to initialize CUDA miner");
    }
}
