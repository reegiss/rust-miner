use rust_miner::mining::nbits_to_target;

#[test]
fn test_nbits_conversion() {
    // Test case: nBits = 0x1a024c62 (difficulty 5)
    // Expected target (little-endian array) has mantissa bytes at offset exponent-3
    let nbits = 0x1a024c62;
    let target = nbits_to_target(nbits);
    
    println!("nBits: 0x{:08x}", nbits);
    println!("Target: {}", hex::encode(&target));
    
    // Exponent = 0x1a = 26 → offset = exponent - 3 = 23
    assert_eq!(target[23], 0x62);
    assert_eq!(target[24], 0x4c);
    assert_eq!(target[25], 0x02);

    // All bytes outside the mantissa window should be zero
    for i in 0..23 {
        assert_eq!(target[i], 0x00);
    }
    for i in 26..32 {
        assert_eq!(target[i], 0x00);
    }
}

#[test]
fn test_difficulty_5_target() {
    // Difficulty 5 with nBits = 0x1a024c62
    let nbits = 0x1a024c62;
    let target = nbits_to_target(nbits);
    
    // A valid share should have hash < target
    // With difficulty 5, we expect roughly 1 share per 5 * 2^32 hashes
    // That's approximately 1 share per 21.5 billion hashes
    
    println!("\nDifficulty 5 target:");
    println!("{}", hex::encode(&target));
    
    // The target (little-endian) has many leading zero bytes
    assert!(target.iter().take(23).all(|&b| b == 0));
}
