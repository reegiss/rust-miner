/// QHash algorithm backend implementation using CUDA
/// 
/// This module wraps CudaMiner to implement the MiningBackend trait,
/// enabling dynamic algorithm dispatch.

use anyhow::Result;
use crate::backend::{MiningBackend, MiningResult, GpuInfo};
use crate::stratum::StratumJob;
use crate::mining::{calculate_merkle_root, nbits_to_target, hex_to_u32_le, hex_to_bytes_be};
use super::CudaMiner;

/// QHash mining backend using CUDA
pub struct QHashCudaBackend {
    miner: CudaMiner,
}

impl QHashCudaBackend {
    /// Create new QHash CUDA backend
    pub fn new() -> Result<Self> {
        let miner = CudaMiner::new()?;
        Ok(Self { miner })
    }
}

impl MiningBackend for QHashCudaBackend {
    fn initialize(&mut self) -> Result<()> {
        // CUDA initialization already done in CudaMiner::new()
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
        // Parse ntime for QHash
        let ntime = hex_to_u32_le(&job.ntime)?;
        
        // Calculate target from nBits (big-endian)
        let nbits_bytes = hex_to_bytes_be(&job.nbits)?;
        if nbits_bytes.len() != 4 {
            anyhow::bail!("Expected 4 bytes for nbits, got {}", nbits_bytes.len());
        }
        let nbits = u32::from_be_bytes([nbits_bytes[0], nbits_bytes[1], nbits_bytes[2], nbits_bytes[3]]);
        let target = nbits_to_target(nbits);
        
    // Calculate merkle root (SHA256d output is big-endian) and convert to little-endian for header
    let mut merkle_root = calculate_merkle_root(job, extranonce1, extranonce2)?;
    merkle_root.reverse(); // Critical: block header stores merkle root little-endian
        
        // Build block header WITHOUT nonce (76 bytes)
        let mut header_76 = [0u8; 76];
        
        // Version (4 bytes, little-endian)
        let version = hex_to_u32_le(&job.version)?;
        header_76[0..4].copy_from_slice(&version.to_le_bytes());
        
        // Previous block hash (32 bytes, need to convert from hex to little-endian)
        let prevhash_hex = &job.prevhash;
        let prevhash_bytes = hex::decode(prevhash_hex)?;
        if prevhash_bytes.len() != 32 {
            anyhow::bail!("Invalid prevhash length: {}", prevhash_bytes.len());
        }
        // Prevhash is sent as big-endian hex string, reverse to little-endian
        let mut prevhash = [0u8; 32];
        for (i, &byte) in prevhash_bytes.iter().enumerate() {
            prevhash[31 - i] = byte;
        }
        header_76[4..36].copy_from_slice(&prevhash);
        
    // Merkle root (32 bytes, now little-endian after reverse)
    header_76[36..68].copy_from_slice(&merkle_root);
        
        // nTime (4 bytes, little-endian)
        header_76[68..72].copy_from_slice(&ntime.to_le_bytes());
        
        // nBits (4 bytes, little-endian)
        header_76[72..76].copy_from_slice(&nbits.to_le_bytes());
        
        // Mine on GPU
        let result = self.miner.mine_job(&header_76, ntime, &target, start_nonce, num_nonces)?;
        
        // Convert old format to new MiningResult
        Ok(MiningResult {
            found_share: result.is_some(),
            nonce: result.as_ref().map(|(n, _)| *n),
            hash: result.as_ref().map(|(_, h)| Box::new(*h)),
            hashes_computed: (num_nonces as u64),
            kernel_time_ms: self.last_kernel_ms() as u32,
        })
    }
    
    fn algorithm_name(&self) -> &str {
        "qhash"
    }
    
    fn device_info(&self) -> Result<GpuInfo> {
        Ok(GpuInfo {
            name: self.miner.device_name()?,
            compute_capability: self.miner.compute_capability()?,
            memory_mb: 0, // TODO: query from CUDA
            compute_units: 0, // TODO: query from CUDA
            clock_mhz: 0, // TODO: query from CUDA
        })
    }
    
    fn last_kernel_ms(&self) -> u64 {
        self.miner.last_kernel_ms()
    }
}
