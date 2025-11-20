use rust_miner::mining::nbits_to_target;

#[test]
fn test_target_calculation() {
    // From pool: target = 0x1a020bcf
    let nbits = 0x1a020bcfu32;
    
    let target = nbits_to_target(nbits);
    
    println!("nbits: 0x{:08x}", nbits);
    println!("target bytes: {}", hex::encode(&target));
    
    // Expected format for big-endian target:
    // exponent = 0x1a = 26
    // mantissa = 0x020bcf
    // offset = 32 - 26 = 6
    // So: [0, 0, 0, 0, 0, 0, 2, b, c, f, 0, 0, ...]
    
    let exponent = (nbits >> 24) as usize;
    let mantissa = nbits & 0x00FFFFFF;
    
    println!("exponent: {} (0x{:02x})", exponent, exponent);
    println!("mantissa: 0x{:06x}", mantissa);
    
    // Manual calculation
    let offset = 32 - exponent;
    println!("offset: {}", offset);
    println!("target[{}] should be: 0x{:02x}", offset, (mantissa >> 16) as u8);
    println!("target[{}] should be: 0x{:02x}", offset + 1, (mantissa >> 8) as u8);
    println!("target[{}] should be: 0x{:02x}", offset + 2, mantissa as u8);
    
    // Print target with indices
    for (i, &byte) in target.iter().enumerate() {
        if i == offset || i == offset + 1 || i == offset + 2 {
            println!("target[{}]: 0x{:02x} ***", i, byte);
        } else {
            println!("target[{}]: 0x{:02x}", i, byte);
        }
    }
    
    // Verify the calculation
    assert_eq!(target[offset], (mantissa >> 16) as u8);
    assert_eq!(target[offset + 1], (mantissa >> 8) as u8);
    assert_eq!(target[offset + 2], mantissa as u8);
}

#[test]
fn test_bitcoin_genesis_target() {
    // Bitcoin genesis block: 0x1d00ffff
    let nbits = 0x1d00ffffu32;
    let target = nbits_to_target(nbits);
    
    println!("\n=== Bitcoin Genesis ===");
    println!("nbits: 0x{:08x}", nbits);
    println!("target: {}", hex::encode(&target));
    
    // Expected: 0x00000000ffff0000000000000000000000000000000000000000000000000000
    // exponent = 0x1d = 29
    // mantissa = 0x00ffff
    // offset = 32 - 29 = 3
    // So: [0, 0, 0, 0, f, f, f, f, 0, 0, ...]
}

#[test]
fn test_hash_vs_target_comparison() {
    // Example hashes and targets
    
    // Target from pool test
    let nbits = 0x1a020bcfu32;
    let target = nbits_to_target(nbits);
    
    println!("\n=== Hash Comparison ===");
    println!("target: {}", hex::encode(&target));
    
    // Create a test hash that is less than target (should be valid)
    // Target is: 000000000000020bcf00...
    // Valid hash: 000000000000020bce00... (< target at byte 8)
    let mut valid_hash = [0u8; 32];
    valid_hash[6] = 0x02;
    valid_hash[7] = 0x0b;
    valid_hash[8] = 0xce; // < 0xcf at position 8
    println!("valid_hash (should be < target): {}", hex::encode(&valid_hash));
    
    // Create a test hash that is greater than target (should be invalid)
    // Invalid hash: 000000000000020bd000... (> target at byte 8)
    let mut invalid_hash = [0u8; 32];
    invalid_hash[6] = 0x02;
    invalid_hash[7] = 0x0b;
    invalid_hash[8] = 0xd0; // > 0xcf at position 8
    println!("invalid_hash (should be > target): {}", hex::encode(&invalid_hash));
    
    // Big-endian comparison logic (as in kernel)
    let mut valid_meets_target = true;
    for i in 0..32 {
        if valid_hash[i] < target[i] {
            break; // hash < target, valid!
        } else if valid_hash[i] > target[i] {
            valid_meets_target = false;
            break;
        }
    }
    
    let mut invalid_meets_target = true;
    for i in 0..32 {
        if invalid_hash[i] < target[i] {
            break; // hash < target, valid!
        } else if invalid_hash[i] > target[i] {
            invalid_meets_target = false;
            break;
        }
    }
    
    println!("valid_hash meets target: {}", valid_meets_target);
    println!("invalid_hash meets target: {}", invalid_meets_target);
    
    assert!(valid_meets_target, "Valid hash should meet target");
    assert!(!invalid_meets_target, "Invalid hash should not meet target");
}
