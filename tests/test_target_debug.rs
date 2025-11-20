use rust_miner::mining::nbits_to_target;

#[test]
fn test_target_calculation() {
    // From pool: target = 0x1a020bcf
    let nbits = 0x1a020bcfu32;
    
    let target = nbits_to_target(nbits);
    
    println!("nbits: 0x{:08x}", nbits);
    println!("target bytes: {}", hex::encode(&target));
    
    // Our nbits_to_target returns little-endian bytes (like Bitcoin's arith_uint256).
    // exponent = 0x1a = 26
    // mantissa = 0x020bcf
    // offset = exponent - 3 = 23
    // So: bytes 23..=25 contain [0xcf, 0x0b, 0x02] (little-endian mantissa)
    
    let exponent = (nbits >> 24) as usize;
    let mantissa = nbits & 0x00FFFFFF;
    
    println!("exponent: {} (0x{:02x})", exponent, exponent);
    println!("mantissa: 0x{:06x}", mantissa);
    
    // Manual calculation
    let offset = exponent - 3;
    println!("offset: {}", offset);
    println!("target[{}] should be: 0x{:02x}", offset, (mantissa & 0xFF) as u8);
    println!("target[{}] should be: 0x{:02x}", offset + 1, ((mantissa >> 8) & 0xFF) as u8);
    println!("target[{}] should be: 0x{:02x}", offset + 2, ((mantissa >> 16) & 0xFF) as u8);

    for (i, &byte) in target.iter().enumerate() {
        if i == offset || i == offset + 1 || i == offset + 2 {
            println!("target[{}]: 0x{:02x} ***", i, byte);
        } else {
            println!("target[{}]: 0x{:02x}", i, byte);
        }
    }

    assert_eq!(target[offset], (mantissa & 0xFF) as u8);
    assert_eq!(target[offset + 1], ((mantissa >> 8) & 0xFF) as u8);
    assert_eq!(target[offset + 2], ((mantissa >> 16) & 0xFF) as u8);
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

    let mut valid_hash = target;
    if valid_hash[25] > 0 {
        valid_hash[25] -= 1;
    } else {
        valid_hash[24] = valid_hash[24].saturating_sub(1);
    }
    println!("valid_hash (should be < target): {}", hex::encode(&valid_hash));

    let mut invalid_hash = target;
    invalid_hash[25] = invalid_hash[25].saturating_add(1);
    println!("invalid_hash (should be > target): {}", hex::encode(&invalid_hash));

    let mut valid_meets_target = true;
    for i in (0..32).rev() {
        if valid_hash[i] < target[i] {
            break;
        } else if valid_hash[i] > target[i] {
            valid_meets_target = false;
            break;
        }
    }

    let mut invalid_meets_target = true;
    for i in (0..32).rev() {
        if invalid_hash[i] < target[i] {
            break;
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
