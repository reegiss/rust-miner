use anyhow::{anyhow, Result};
use cudarc::driver::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering, AtomicUsize};
use std::sync::Mutex;
use cudarc::driver::sys::lib;
use cudarc::driver::sys::CUresult;

/// Ethash CUDA kernel source code
const ETHASH_KERNEL_SRC: &str = include_str!("ethash.cu");

/// Internal Ethash CUDA miner implementation
pub struct EthashCudaMiner {
    device: Arc<CudaDevice>,
    func: CudaFunction,
    func_search: Option<CudaFunction>,
    d_dag: Arc<Mutex<Option<CudaSlice<u8>>>>,
    dag_bytes: AtomicUsize,
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
        device.load_ptx(ptx, "ethash", &["ethash_mine", "ethash_search"])?;
        let func = device.get_func("ethash", "ethash_mine")
            .ok_or_else(|| anyhow!("Failed to load ethash_mine function"))?;
        let func_search = device.get_func("ethash", "ethash_search");
            
        Ok(Self {
            device,
            func,
            func_search,
            d_dag: Arc::new(Mutex::new(None)),
            dag_bytes: AtomicUsize::new(0),
            last_kernel_ms: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Ensure DAG is loaded in VRAM for this device
    pub fn ensure_dag_loaded(&self, dataset_path: &std::path::Path) -> Result<()> {
        // If already loaded and size matches file, skip reload
        let meta = std::fs::metadata(dataset_path)?;
        let dag_bytes = meta.len() as usize;
        if self.d_dag.lock().unwrap().is_some() && self.dag_bytes.load(Ordering::Relaxed) == dag_bytes {
            return Ok(());
        }
        tracing::info!("Loading DAG into VRAM: {} ({} MB)", dataset_path.display(), (dag_bytes as f64)/(1024.0*1024.0));
        let data = std::fs::read(dataset_path)?; // NOTE: may be large; consider chunked uploads in future
        let d = self.device.htod_copy(data)?;
        *self.d_dag.lock().unwrap() = Some(d);
        self.dag_bytes.store(dag_bytes, Ordering::Relaxed);
        Ok(())
    }
    
    /// Mine a job on GPU
    pub fn mine_job(
        &self,
        header_without_nonce: &[u8; 76], // 76 bytes (header minus nonce)
        start_nonce: u32,
        num_nonces: u32,
        difficulty_target: u32, // Simplified target for now
    ) -> Result<Option<(u32, [u8; 32])>> {
        // Require DAG preloaded when using the ethash_search kernel
        let use_search = self.func_search.is_some();
        if use_search && self.d_dag.lock().unwrap().is_none() {
            return Err(anyhow!("DAG not loaded in VRAM for Ethash"));
        }
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

        // Found hash output buffer (32 bytes)
        let h_found_hash = vec![0u8; 32];
        let d_found_hash = self.device.htod_copy(h_found_hash.clone())?;
        
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
            if use_search {
                let guard = self.d_dag.lock().unwrap();
                let dag = guard.as_ref().expect("checked above to be Some");
                let dag_bytes = self.dag_bytes.load(Ordering::Relaxed) as u64;
                // Prefer ethash_search if available
                self.func_search.clone().unwrap()
                    .launch(
                        cfg,
                        (
                            dag,
                            dag_bytes,
                            &d_header,
                            start_nonce,
                            num_nonces,
                            &d_found_hash, // reuse buffer as target placeholder (unused in simplified)
                            &d_solution,
                            &d_found_hash,
                        ),
                    )
                    .map_err(|e| anyhow!("Kernel launch failed: {}", e))?;
            } else {
                self.func.clone()
                    .launch(
                        cfg,
                        (
                            &d_header,
                            start_nonce,
                            &d_solution,
                            difficulty_target,
                            &d_found_hash,
                        ),
                    )
                    .map_err(|e| anyhow!("Kernel launch failed: {}", e))?;
            }
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
            let mut h_found_hash_out = vec![0u8; 32];
            self.device.dtoh_sync_copy_into(&d_found_hash, &mut h_found_hash_out)?;
            let found_hash_arr: [u8; 32] = h_found_hash_out.try_into().map_err(|_| anyhow!("unexpected hash length"))?;
            Ok(Some((h_solution_result[0], found_hash_arr)))
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
