use crate::algorithms::{HashAlgorithm, QHash};
use crate::stratum::protocol::StratumJob;
use anyhow::{Result, anyhow};
use sha2::{Sha256, Digest};

/// Mining statistics
#[derive(Debug, Clone)]
pub struct MiningStats {
    pub hashes: u64,
    pub shares_found: u64,
    pub shares_accepted: u64,
    pub shares_rejected: u64,
}

impl MiningStats {
    pub fn new() -> Self {
        Self {
            hashes: 0,
            shares_found: 0,
            shares_accepted: 0,
            shares_rejected: 0,
        }
    }
}

/// Convert hex string to bytes (reverse for little-endian)
fn hex_to_bytes_le(hex: &str) -> Result<Vec<u8>> {
    let hex = hex.trim_start_matches("0x");
    let bytes = hex::decode(hex)?;
    Ok(bytes.into_iter().rev().collect())
}

/// Convert hex string to bytes (as-is for big-endian)
fn hex_to_bytes_be(hex: &str) -> Result<Vec<u8>> {
    let hex = hex.trim_start_matches("0x");
    Ok(hex::decode(hex)?)
}

/// Convert hex string to u32 (little-endian)
fn hex_to_u32_le(hex: &str) -> Result<u32> {
    let bytes = hex_to_bytes_be(hex)?;
    if bytes.len() != 4 {
        return Err(anyhow!("Expected 4 bytes for u32, got {}", bytes.len()));
    }
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

/// Build block header from Stratum job
/// 
/// Block header structure (80 bytes):
/// - version (4 bytes)
/// - prevhash (32 bytes)
/// - merkle_root (32 bytes)
/// - ntime (4 bytes)
/// - nbits (4 bytes)
/// - nonce (4 bytes)
pub fn build_block_header(
    job: &StratumJob,
    extranonce1: &str,
    extranonce2: &[u8],
    nonce: u32,
) -> Result<[u8; 80]> {
    let mut header = [0u8; 80];
    let mut offset = 0;
    
    // 1. Version (4 bytes, little-endian)
    let version = hex_to_u32_le(&job.version)?;
    header[offset..offset + 4].copy_from_slice(&version.to_le_bytes());
    offset += 4;
    
    // 2. Previous block hash (32 bytes, little-endian)
    let prevhash = hex_to_bytes_le(&job.prevhash)?;
    if prevhash.len() != 32 {
        return Err(anyhow!("Invalid prevhash length: {}", prevhash.len()));
    }
    header[offset..offset + 32].copy_from_slice(&prevhash);
    offset += 32;
    
    // 3. Merkle root (32 bytes) - needs to be calculated from coinbase tx
    let merkle_root = calculate_merkle_root(job, extranonce1, extranonce2)?;
    header[offset..offset + 32].copy_from_slice(&merkle_root);
    offset += 32;
    
    // 4. nTime (4 bytes, little-endian)
    let ntime = hex_to_u32_le(&job.ntime)?;
    header[offset..offset + 4].copy_from_slice(&ntime.to_le_bytes());
    offset += 4;
    
    // 5. nBits (4 bytes, little-endian)
    let nbits = hex_to_u32_le(&job.nbits)?;
    header[offset..offset + 4].copy_from_slice(&nbits.to_le_bytes());
    offset += 4;
    
    // 6. Nonce (4 bytes, little-endian)
    header[offset..offset + 4].copy_from_slice(&nonce.to_le_bytes());
    
    Ok(header)
}

/// Calculate merkle root from coinbase transaction
fn calculate_merkle_root(
    job: &StratumJob,
    extranonce1: &str,
    extranonce2: &[u8],
) -> Result<[u8; 32]> {
    // Build coinbase transaction
    let coinb1 = hex::decode(&job.coinb1)?;
    let coinb2 = hex::decode(&job.coinb2)?;
    let en1 = hex::decode(extranonce1)?;
    
    let mut coinbase_tx = Vec::new();
    coinbase_tx.extend_from_slice(&coinb1);
    coinbase_tx.extend_from_slice(&en1);
    coinbase_tx.extend_from_slice(extranonce2);
    coinbase_tx.extend_from_slice(&coinb2);
    
    // Double SHA256 of coinbase transaction
    let mut hasher = Sha256::new();
    hasher.update(&coinbase_tx);
    let hash1 = hasher.finalize();
    
    let mut hasher = Sha256::new();
    hasher.update(hash1);
    let mut merkle_root: [u8; 32] = hasher.finalize().into();
    
    // Apply merkle branches
    for branch_hex in &job.merkle_branch {
        let branch = hex::decode(branch_hex)?;
        if branch.len() != 32 {
            return Err(anyhow!("Invalid merkle branch length: {}", branch.len()));
        }
        
        // Concatenate and double SHA256
        let mut hasher = Sha256::new();
        hasher.update(&merkle_root);
        hasher.update(&branch);
        let hash1 = hasher.finalize();
        
        let mut hasher = Sha256::new();
        hasher.update(hash1);
        merkle_root = hasher.finalize().into();
    }
    
    Ok(merkle_root)
}

/// Convert nBits compact format to full 256-bit target
pub fn nbits_to_target(nbits: u32) -> [u8; 32] {
    let mut target = [0u8; 32];
    
    let exponent = (nbits >> 24) as usize;
    let mantissa = nbits & 0x00FFFFFF;
    
    if exponent <= 3 {
        let shifted = mantissa >> (8 * (3 - exponent));
        target[29] = (shifted >> 16) as u8;
        target[30] = (shifted >> 8) as u8;
        target[31] = shifted as u8;
    } else {
        let offset = 32 - exponent;
        if offset < 29 {
            target[offset] = (mantissa >> 16) as u8;
            target[offset + 1] = (mantissa >> 8) as u8;
            target[offset + 2] = mantissa as u8;
        }
    }
    
    target
}

/// Mine a single job (CPU version)
pub fn mine_job_cpu(
    job: &StratumJob,
    extranonce1: &str,
    extranonce2: &[u8],
    start_nonce: u32,
    end_nonce: u32,
    stats: &mut MiningStats,
) -> Result<Option<(u32, [u8; 32])>> {
    // Parse ntime for QHash
    let ntime = hex_to_u32_le(&job.ntime)?;
    let qhash = QHash::new(ntime);
    
    // Calculate target from nBits
    let nbits = hex_to_u32_le(&job.nbits)?;
    let target = nbits_to_target(nbits);
    
    // Mine nonce range
    for nonce in start_nonce..end_nonce {
        // Build block header with current nonce
        let header = build_block_header(job, extranonce1, extranonce2, nonce)?;
        
        // Calculate hash
        let hash = qhash.hash(&header);
        stats.hashes += 1;
        
        // Check if hash meets target
        if qhash.meets_target(&hash, &target) {
            stats.shares_found += 1;
            tracing::info!(
                "Found share! Nonce: 0x{:08x}, Hash: {}",
                nonce,
                hex::encode(hash)
            );
            return Ok(Some((nonce, hash)));
        }
        
        // Log progress every 100k hashes
        if stats.hashes % 100_000 == 0 {
            tracing::debug!(
                "Mining progress: {} hashes, nonce: 0x{:08x}",
                stats.hashes,
                nonce
            );
        }
    }
    
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_nbits_to_target() {
        // Example: 0x1d00ffff (Bitcoin genesis block difficulty)
        let nbits = 0x1d00ffff;
        let target = nbits_to_target(nbits);
        
        // Target should start with zeros
        assert_eq!(target[0], 0x00);
        assert_eq!(target[1], 0x00);
        assert_eq!(target[2], 0x00);
        
        println!("Target: {}", hex::encode(target));
    }
    
    #[test]
    fn test_hex_conversions() {
        let hex = "01000000";
        let le = hex_to_bytes_le(hex).unwrap();
        assert_eq!(le, vec![0x00, 0x00, 0x00, 0x01]);
        
        let u32_val = hex_to_u32_le(hex).unwrap();
        assert_eq!(u32_val, 1);
    }
}
