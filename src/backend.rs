/// Mining backend trait for algorithm abstraction
/// 
/// This trait defines the interface that all mining backends must implement.
/// It enables dynamic dispatch of different algorithms (qhash, sha256d, etc.)
/// while maintaining a common interface for the mining loop.

use anyhow::Result;
use crate::stratum::StratumJob;

/// Mining backend trait - all algorithms must implement this
pub trait MiningBackend: Send + Sync {
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
    /// * `Ok(Some((nonce, hash)))` - Solution found
    /// * `Ok(None)` - No solution in this range
    /// * `Err(_)` - Mining error occurred
    fn mine_job(
        &self,
        job: &StratumJob,
        extranonce1: &str,
        extranonce2: &[u8],
        start_nonce: u32,
        num_nonces: u32,
    ) -> Result<Option<(u32, [u8; 32])>>;
    
    /// Get the algorithm name (e.g., "qhash", "sha256d")
    fn _algorithm_name(&self) -> &str;
    
    /// Get device name (e.g., "NVIDIA GeForce GTX 1660 SUPER")
    fn device_name(&self) -> Result<String>;
    
    /// Get compute capability (major, minor) for CUDA devices
    fn compute_capability(&self) -> Result<(i32, i32)>;
    
    /// Get last kernel execution time in milliseconds (for adaptive batching)
    fn last_kernel_ms(&self) -> u64;
}
