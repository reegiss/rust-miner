use anyhow::{anyhow, Result};
use cudarc::driver::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use cudarc::driver::sys::lib;
use cudarc::driver::sys::CUresult;

/// Ethash CUDA kernel source code
const ETHASH_KERNEL_SRC: &str = include_str!("ethash.cu");

/// Internal Ethash CUDA miner implementation
#[derive(Clone)]
pub struct EthashCudaMiner {
    device: Arc<CudaDevice>,
    func: CudaFunction,
    last_kernel_ms: Arc<AtomicU64>,
}

impl EthashCudaMiner {
    /// Create new Ethash CUDA miner for specific device
    pub fn new(device_index: usize) -> Result<Self> {
        // Set CUDA device
        unsafe {
            cudarc::driver::sys::lib().cuDevicePrimaryCtxRetain(
                std::ptr::null_mut(),
                device_index as i32
            );
            cudarc::driver::sys::lib().cuCtxSetCurrent(std::ptr::null_mut());
        }
        
        // Initialize CUDA device
        let device = CudaDevice::new(device_index)?;
        
        tracing::info!("Initializing Ethash CUDA miner...");
        
        // Compile CUDA kernel
        tracing::info!("Compiling Ethash CUDA kernel...");
        let ptx = cudarc::nvrtc::compile_ptx(ETHASH_KERNEL_SRC)
            .map_err(|e| anyhow!("Failed to compile Ethash CUDA kernel: {}", e))?;
        
        tracing::info!("Loading Ethash CUDA module...");
        
        // Load module and get function
        device.load_ptx(ptx, "ethash", &["ethash_mine"])?;
        let func = device.get_func("ethash", "ethash_mine")
            .ok_or_else(|| anyhow!("Failed to load ethash_mine function"))?;
            
        // TODO: Allocate DAG memory here (3.6GB+)
        // For now, we run without DAG (simplified kernel)
        
        Ok(Self {
            device,
            func,
            last_kernel_ms: Arc::new(AtomicU64::new(0)),
        })
    }
    
    /// Mine a job on GPU
    pub fn mine_job(
        &self,
        header_without_nonce: &[u8; 76], // 76 bytes (header minus nonce)
        start_nonce: u32,
        num_nonces: u32,
        difficulty_target: u32, // Simplified target for now
    ) -> Result<Option<u32>> {
        // Prepare header (80 bytes expected by kernel, but we pass 76 and it handles nonce)
        // Actually kernel expects 80 bytes. We should construct the full 80 bytes with 0 nonce first.
        let mut full_header = [0u8; 80];
        full_header[0..76].copy_from_slice(header_without_nonce);
        // Last 4 bytes are nonce, will be set by kernel threads
        
        // Allocate device memory
        let d_header = self.device.htod_copy(full_header.to_vec())?;
        
        // Solution buffer: [nonce, found_flag]
        let h_solution = vec![0u32, 0u32];
        let d_solution = self.device.htod_copy(h_solution.clone())?;
        
        // Launch configuration
    let threads_per_block: u32 = 256;
    let num_blocks = num_nonces.div_ceil(threads_per_block);
        
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };
        
        // tracing::trace!("Launching kernel: {} blocks, {} threads", num_blocks, threads_per_block);

        unsafe {
            self.func.clone()
                .launch(
                    cfg,
                    (
                        &d_header,
                        start_nonce,
                        &d_solution,
                        difficulty_target,
                    ),
                )
                .map_err(|e| anyhow!("Kernel launch failed: {}", e))?;
        }
        
        // Poll for completion
        let stream = *self.device.cu_stream();
        let poll_start = std::time::Instant::now();
        let mut polls = 0;
        loop {
            let result = unsafe { lib().cuStreamQuery(stream) };
            match result {
                CUresult::CUDA_SUCCESS => {
                    let elapsed = poll_start.elapsed();
                    let ms = (elapsed.as_secs_f64() * 1000.0) as u64;
                    self.last_kernel_ms.store(ms, Ordering::Relaxed);
                    // tracing::trace!("Kernel finished in {} ms ({} polls)", ms, polls);
                    break;
                }
                CUresult::CUDA_ERROR_NOT_READY => {
                    polls += 1;
                    if polls % 1000 == 0 {
                         // tracing::trace!("Kernel still running... {} polls", polls);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                _ => return Err(anyhow!("Stream query failed: {:?}", result)),
            }
        }
        
        // Check results
        let mut h_solution_result = vec![0u32; 2];
        self.device.dtoh_sync_copy_into(&d_solution, &mut h_solution_result)?;
        
        if h_solution_result[1] == 1 {
            Ok(Some(h_solution_result[0]))
        } else {
            Ok(None)
        }
    }
    
    pub fn device_name(&self) -> Result<String> {
        Ok(self.device.name()?)
    }
    
    pub fn compute_capability(&self) -> Result<(i32, i32)> {
        use cudarc::driver::sys::CUdevice_attribute;
        let major = self.device.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor = self.device.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        Ok((major, minor))
    }

    pub fn last_kernel_ms(&self) -> u64 {
        self.last_kernel_ms.load(Ordering::Relaxed)
    }
}
