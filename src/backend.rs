/// GPU Mining backend abstraction layer (Hexagonal Architecture - Port)
/// Enables swapping GPU implementations (CUDA-only) and algorithms

use anyhow::Result;
use crate::stratum::StratumJob;

/// Result of a mining attempt
#[derive(Clone, Debug)]
pub struct MiningResult {
    pub found_share: bool,
    pub nonce: Option<u32>,
    pub hash: Option<Box<[u8; 32]>>,
    pub hashes_computed: u64,
    pub kernel_time_ms: u32,
}

/// GPU device information
#[derive(Clone, Debug)]
pub struct GpuInfo {
    pub name: String,
    pub compute_capability: (i32, i32),
    pub memory_mb: u32,
    pub compute_units: u32,
    pub clock_mhz: u32,
}

/// Mining backend trait - abstract interface for GPU mining
/// All GPU backends (CUDA, future OpenCL/HIP) must implement this
pub trait MiningBackend: Send + Sync {
    /// Initialize GPU and allocate resources
    fn initialize(&mut self) -> Result<()>;
    
    /// Mine a job with the given parameters
    /// 
    /// # Arguments
    /// * `job` - Stratum job from pool
    /// * `extranonce1` - Pool-provided extranonce1 (hex string)
    /// * `extranonce2` - Miner-generated extranonce2 (raw bytes)
    /// * `start_nonce` - Starting nonce for this batch
    /// * `num_nonces` - Number of nonces to try in this batch
    /// 
    /// # Returns
    /// Mining result with found nonce (if any) and statistics
    fn mine_job(
        &self,
        job: &StratumJob,
        extranonce1: &str,
        extranonce2: &[u8],
        start_nonce: u32,
        num_nonces: u32,
    ) -> Result<MiningResult>;
    
    /// Get the algorithm name (e.g., "qhash", "sha256d")
    fn algorithm_name(&self) -> &str;
    
    /// Get GPU device info
    fn device_info(&self) -> Result<GpuInfo>;
    
    /// Get last kernel execution time in milliseconds (for monitoring)
    fn last_kernel_ms(&self) -> u64;
}
