
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

/// Convert hex string to big-endian bytes
pub fn hex_to_bytes_be(hex: &str) -> Result<Vec<u8>> {
    let hex = hex.trim_start_matches("0x");
    Ok(hex::decode(hex)?)
}

/// Convert hex string to little-endian u32
pub fn hex_to_u32_le(hex: &str) -> Result<u32> {
    let bytes = hex_to_bytes_be(hex)?;
    if bytes.len() != 4 {
        return Err(anyhow!("Expected 4 bytes for u32, got {}", bytes.len()));
    }
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

/// Calculate merkle root from coinbase transaction
pub fn calculate_merkle_root(
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
        // Small exponent: shift right
        let shifted = mantissa >> (8 * (3 - exponent));
        target[29] = (shifted >> 16) as u8;
        target[30] = (shifted >> 8) as u8;
        target[31] = shifted as u8;
    } else if exponent <= 32 {
        // Normal range: place mantissa at correct position
        let offset = 32 - exponent;
        target[offset] = (mantissa >> 16) as u8;
        if offset + 1 < 32 {
            target[offset + 1] = (mantissa >> 8) as u8;
        }
        if offset + 2 < 32 {
            target[offset + 2] = mantissa as u8;
        }
    } else {
        // Exponent > 32: target is larger than 256 bits, return max
        target.fill(0xFF);
    }
    
    target
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
