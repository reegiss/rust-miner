/// CUDA mining implementation for qhash
use anyhow::{anyhow, Result};
use cudarc::driver::*;
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

// Import raw CUDA API for non-blocking polling
use cudarc::driver::sys::lib;
use cudarc::driver::sys::{CUstream, CUresult};

const CUDA_KERNEL_SRC: &str = include_str!("qhash.cu");

#[derive(Clone)]
pub struct CudaMiner {
    device: Arc<CudaDevice>,
    func: CudaFunction,
}

impl CudaMiner {
    /// Create new CUDA miner
    pub fn new() -> Result<Self> {
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
        })
    }
    
    /// Mine a job on GPU
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
        
        // Launch configuration
        // Note: qhash kernel is complex and uses many registers
        // Using 256 threads/block for better register allocation
        let threads_per_block = 256;
        let num_blocks = (num_nonces + threads_per_block - 1) / threads_per_block;
        
        tracing::debug!(
            "Launching kernel: {} blocks x {} threads = {} nonces",
            num_blocks,
            threads_per_block,
            num_nonces
        );
        
        // Launch kernel
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
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
                        num_nonces,
                    ),
                )
                .map_err(|e| anyhow!("Kernel launch failed: {}", e))?;
        }
        
        // Use NON-BLOCKING polling instead of synchronize() to reduce CPU usage
        // This is how miners like wildrig-multi achieve <1% CPU usage
        let stream = unsafe { *self.device.cu_stream() };
        
        loop {
            let result = unsafe { lib().cuStreamQuery(stream) };
            
            match result {
                CUresult::CUDA_SUCCESS => {
                    // Kernel finished
                    break;
                }
                CUresult::CUDA_ERROR_NOT_READY => {
                    // Kernel still running - sleep to avoid busy-wait
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    continue;
                }
                _ => {
                    // Error occurred
                    return Err(anyhow!("Stream query failed: {:?}", result));
                }
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
            // Solution found! Compute the hash on CPU for verification
            let hash = self.compute_hash_cpu(block_header, found_nonce, ntime)?;
            Ok(Some((found_nonce, hash)))
        }
    }
    
    /// Compute qhash on CPU (for verification)
    fn compute_hash_cpu(
        &self,
        block_header: &[u8; 76],
        nonce: u32,
        ntime: u32,
    ) -> Result<[u8; 32]> {
        use crate::algorithms::qhash::QHash;
        use crate::algorithms::HashAlgorithm;
        
        let mut header = [0u8; 80];
        header[..76].copy_from_slice(block_header);
        header[76..80].copy_from_slice(&nonce.to_le_bytes());
        
        let qhash = QHash::new(ntime);
        Ok(qhash.hash(&header))
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
