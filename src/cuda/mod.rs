/// CUDA mining implementation
mod qhash_backend;
mod ethash_backend;
mod ethash_miner;

pub use qhash_backend::QHashCudaBackend;
pub use ethash_backend::EthashCudaBackend;
pub use ethash_miner::EthashCudaMiner;

use anyhow::{anyhow, Result};
use cudarc::driver::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

// Import raw CUDA API for non-blocking polling
use cudarc::driver::sys::lib;
use cudarc::driver::sys::CUresult;

/// CUDA kernel source code (included at compile time)
const CUDA_KERNEL_SRC: &str = include_str!("qhash.cu");

/// Set CUDA device for current thread
pub fn set_cuda_device(device_index: usize) -> Result<()> {
    unsafe {
        cudarc::driver::sys::lib().cuDevicePrimaryCtxRetain(
            std::ptr::null_mut(),
            device_index as i32
        );
        cudarc::driver::sys::lib().cuCtxSetCurrent(std::ptr::null_mut());
    }
    Ok(())
}

/// Helper to compile CUDA kernel with aggressive optimizations
/// 
/// Ideally, we would use NVRTC with options:
/// - `-O3`: Maximum optimization level
/// - `--use_fast_math`: Enable fast FP32 math
/// - `--gpu-architecture=compute_75`: Target Turing architecture explicitly
/// 
/// However, cudarc 0.11 doesn't expose option passing to compile_ptx.
/// As a workaround, we:
/// 1. Use compile_ptx with default options
/// 2. Future: Wrap nvrtc_sys directly to support compile options
/// 
/// The kernel is already optimized with:
/// - `#pragma unroll` directives in hot paths
/// - Register allocation optimization via CUDA pragma comments
/// - Fast math-friendly operations
fn compile_optimized_kernel() -> Result<cudarc::nvrtc::Ptx> {
    // Current limitation: cudarc 0.11's compile_ptx doesn't support options.
    // Using default compilation which applies O2 level optimization.
    // TODO (future): Move to nvrtc_sys to enable -O3, --use_fast_math flags for even better performance.
    
    tracing::info!("Compiling CUDA kernel (O2 optimization)");
    
    cudarc::nvrtc::compile_ptx(CUDA_KERNEL_SRC)
        .map_err(|e| anyhow!("Failed to compile CUDA kernel: {}", e))
}

/// Internal CUDA miner implementation (used by backends)
/// Made public for benchmarking
#[derive(Clone)]
pub struct CudaMiner {
    device: Arc<CudaDevice>,
    func: CudaFunction,
    d_lookup_table: CudaSlice<f64>, // Lookup table on GPU
    last_kernel_ms: Arc<AtomicU64>, // duration of last kernel for adaptive polling
}

impl CudaMiner {
    /// Create new CUDA miner for specific device
    pub fn new(device_index: usize) -> Result<Self> {
        // Set CUDA device before creating device context
        set_cuda_device(device_index)?;
        
        // Initialize CUDA device
        let device = CudaDevice::new(device_index)?;
        
        tracing::info!("Initializing CUDA miner...");
        
        // Compile CUDA kernel to PTX with optimization
        let ptx = compile_optimized_kernel()?;
        
        tracing::info!("Loading CUDA module...");
        
        // Load module and get function
        device.load_ptx(ptx, "qhash", &["qhash_mine"])?;
        let func = device.get_func("qhash", "qhash_mine")
            .ok_or_else(|| anyhow!("Failed to load qhash_mine function"))?;
        
        // Load lookup table (65536 doubles = 512 KB)
        tracing::info!("Loading qHash lookup table (512 KB)...");
        let lookup_table_bytes = include_bytes!("qhash_lookup_table.bin");
        
        // Convert bytes to f64 array (65536 entries)
        const LOOKUP_SIZE: usize = 65536;
        let mut lookup_table = vec![0.0f64; LOOKUP_SIZE];

        for (i, slot) in lookup_table.iter_mut().enumerate().take(LOOKUP_SIZE) {
            let offset = i * 8;
            let bytes = [
                lookup_table_bytes[offset],
                lookup_table_bytes[offset + 1],
                lookup_table_bytes[offset + 2],
                lookup_table_bytes[offset + 3],
                lookup_table_bytes[offset + 4],
                lookup_table_bytes[offset + 5],
                lookup_table_bytes[offset + 6],
                lookup_table_bytes[offset + 7],
            ];
            *slot = f64::from_le_bytes(bytes);
        }
        
        // Upload lookup table to GPU (passed as kernel parameter)
        let d_lookup_table = device.htod_copy(lookup_table)?;
        
        tracing::info!("CUDA miner initialized successfully with analytical QHash approximation");
        
        Ok(Self {
            device,
            func,
            d_lookup_table,
            last_kernel_ms: Arc::new(AtomicU64::new(0)),
        })
    }
    
    /// Mine a job on GPU (blocking, low CPU via driver polling + sleep)
    ///
    /// Returns Option<(nonce, hash)> if a valid share is found
    pub fn mine_job(
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
        
        // Launch configuration - testing for occupancy optimization
        // Baseline (Phase 1): 128 threads/block (12.5% occupancy, 37 MH/s, 400ms kernel)
    let threads_per_block: u32 = 256;
    let num_blocks = num_nonces.div_ceil(threads_per_block);
        
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
                        &d_target,
                        start_nonce,
                        num_nonces,
                        &d_solution,
                        &d_found_hash,
                        &self.d_lookup_table,  // Pass lookup table pointer
                    ),
                )
                .map_err(|e| anyhow!("Kernel launch failed: {}", e))?;
        }
        
        // Non-blocking driver polling with adaptive sleep (host thread, not async)
        let stream = *self.device.cu_stream();
        let poll_start = std::time::Instant::now();
        loop {
            let result = unsafe { lib().cuStreamQuery(stream) };
            match result {
                CUresult::CUDA_SUCCESS => {
                    let elapsed = poll_start.elapsed();
                    let ms = (elapsed.as_secs_f64() * 1000.0) as u64;
                    self.last_kernel_ms.store(ms, Ordering::Relaxed);
                    break;
                }
                CUresult::CUDA_ERROR_NOT_READY => {
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
    pub fn device_name(&self) -> Result<String> {
        Ok(self.device.name()?)
    }
    
    /// Get device compute capability
    pub fn compute_capability(&self) -> Result<(i32, i32)> {
        use cudarc::driver::sys::CUdevice_attribute;
        let major = self.device.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor = self.device.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        Ok((major, minor))
    }

    /// Get last kernel duration in milliseconds (0 if none yet)
    pub fn last_kernel_ms(&self) -> u64 {
        self.last_kernel_ms.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_cuda_miner_init() {
        let miner = CudaMiner::new(0);
        assert!(miner.is_ok(), "Failed to initialize CUDA miner");
    }
}
