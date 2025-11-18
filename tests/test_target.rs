use rust_miner::mining::nbits_to_target;

#[test]
fn test_nbits_conversion() {
    // Test case: nBits = 0x1a024c62 (difficulty 5)
    // Expected target = 0x000000024c6200000000000000000000000000000000000000000000000000
    let nbits = 0x1a024c62;
    let target = nbits_to_target(nbits);
    
    println!("nBits: 0x{:08x}", nbits);
    println!("Target: {}", hex::encode(&target));
    
    // Exponent = 0x1a = 26
    // Mantissa = 0x024c62
    // Position = 32 - 26 = 6
    // target[6] = 0x02
    // target[7] = 0x4c
    // target[8] = 0x62
    
    assert_eq!(target[6], 0x02);
    assert_eq!(target[7], 0x4c);
    assert_eq!(target[8], 0x62);
    
    // All bytes before position 6 should be 0
    for i in 0..6 {
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
    
    // The target should start with several zero bytes
    assert!(target[0] == 0 && target[1] == 0 && target[2] == 0);
}
