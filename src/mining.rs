
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
    fn double_sha256(data: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let first = hasher.finalize_reset();
        hasher.update(&first);
        hasher.finalize().into()
    }

    // Build coinbase transaction
    let coinb1 = hex::decode(&job.coinb1)?;
    let coinb2 = hex::decode(&job.coinb2)?;
    let en1 = hex::decode(extranonce1)?;
    
    let mut coinbase_tx = Vec::new();
    coinbase_tx.extend_from_slice(&coinb1);
    coinbase_tx.extend_from_slice(&en1);
    coinbase_tx.extend_from_slice(extranonce2);
    coinbase_tx.extend_from_slice(&coinb2);
    
    // Double SHA256 of coinbase transaction (Stratum expects little-endian for subsequent steps)
    let mut merkle_root = double_sha256(&coinbase_tx);
    merkle_root.reverse();

    // Apply merkle branches (branches are provided little-endian)
    for branch_hex in &job.merkle_branch {
        let branch = hex::decode(branch_hex)?;
        if branch.len() != 32 {
            return Err(anyhow!("Invalid merkle branch length: {}", branch.len()));
        }
        let mut lhs = merkle_root;
        lhs.reverse();

        let mut rhs = [0u8; 32];
        rhs.copy_from_slice(&branch);
        rhs.reverse();

        let mut combined = [0u8; 64];
        combined[..32].copy_from_slice(&lhs);
        combined[32..].copy_from_slice(&rhs);

        merkle_root = double_sha256(&combined);
        merkle_root.reverse();
    }

    merkle_root.reverse();
    Ok(merkle_root)
}

/// Convert nBits compact format to full 256-bit target
/// Returns target in LITTLE-ENDIAN format (like Bitcoin's arith_uint256)
pub fn nbits_to_target(nbits: u32) -> [u8; 32] {
    let mut target = [0u8; 32];
    
    let exponent = (nbits >> 24) as usize;
    let mantissa = nbits & 0x00FFFFFF;
    
    if exponent <= 3 {
        // Small exponent: shift right
        let shifted = mantissa >> (8 * (3 - exponent));
        target[0] = shifted as u8;
        target[1] = (shifted >> 8) as u8;
        target[2] = (shifted >> 16) as u8;
    } else if exponent <= 32 {
        // Normal range: place mantissa at correct position
        // exponent tells us how many bytes from LSB
        let offset = exponent - 3;
        target[offset] = mantissa as u8;
        target[offset + 1] = (mantissa >> 8) as u8;
        target[offset + 2] = (mantissa >> 16) as u8;
    } else {
        // Exponent > 32: target is larger than 256 bits, return max
        target.fill(0xFF);
    }
    
    target
}

/// Calculate difficulty from nBits
/// Difficulty = max_target / current_target
/// Where max_target is 0x00000000FFFF0000000000000000000000000000000000000000000000000000
pub fn nbits_to_difficulty(nbits: u32) -> f64 {
    // Bitcoin's max target (difficulty 1)
    let max_nbits: u32 = 0x1d00ffff;
    let max_target = nbits_to_target(max_nbits);
    let current_target = nbits_to_target(nbits);
    
    // Convert to f64 for division
    // Targets are little-endian, so most significant bytes are at the END
    let mut max_val: u128 = 0;
    let mut cur_val: u128 = 0;
    
    // Read from high bytes (index 31 down) - these are most significant in LE
    for i in 0..16 {
        max_val = (max_val << 8) | (max_target[31 - i] as u128);
        cur_val = (cur_val << 8) | (current_target[31 - i] as u128);
    }
    
    if cur_val == 0 {
        return f64::MAX;
    }
    
    (max_val as f64) / (cur_val as f64)
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
    fn test_nbits_to_difficulty() {
        // Difficulty 1 should return 1.0
        let diff = nbits_to_difficulty(0x1d00ffff);
        assert!((diff - 1.0).abs() < 0.01);
        
        // Higher difficulty should be > 1
        let diff = nbits_to_difficulty(0x1a020722);
        assert!(diff > 1.0);
    }
    
    #[test]
    fn test_hex_conversions() {
        let hex = "01000000";
        let be = hex_to_bytes_be(hex).unwrap();
        assert_eq!(be, vec![0x01, 0x00, 0x00, 0x00]);
        
        let u32_val = hex_to_u32_le(hex).unwrap();
        assert_eq!(u32_val, 1);
    }
}
