fn main() {
    // Difficulty calculation for cryptocurrency mining
    
    // Target for difficulty 1 (maximum target)
    let target_1_hex = "00000000FFFF0000000000000000000000000000000000000000000000000000";
    
    // Target for difficulty 5
    let target_5_hex = "000000000000024c620000000000000000000000000000000000000000000000";
    
    // Convert to numbers for calculation
    // target_1 = 0x00000000FFFF000000...
    // target_5 = 0x000000000000024c6200...
    
    // Probability of finding a valid hash:
    // P = target / 2^256
    
    // For difficulty 5:
    // P ≈ 1 / (difficulty * 2^32)
    // P ≈ 1 / (5 * 4294967296)
    // P ≈ 1 / 21,474,836,480
    
    let difficulty = 5.0;
    let max_32bit = 4294967296.0;
    let prob = 1.0 / (difficulty * max_32bit);
    
    println!("Difficulty: {}", difficulty);
    println!("Probability of success per hash: {:.15}", prob);
    println!("Expected hashes to find one share: {:.0}", 1.0 / prob);
    
    // With hashrate of 20-45 MH/s
    let hashrates = vec![20.0, 30.0, 40.0, 50.0];
    
    println!("\nExpected time to find a share:");
    for &hashrate_mhs in &hashrates {
        let hashrate_hs = hashrate_mhs * 1_000_000.0; // Convert to H/s
        let expected_seconds = (1.0 / prob) / hashrate_hs;
        let expected_minutes = expected_seconds / 60.0;
        
        println!("  At {} MH/s: {:.1} seconds ({:.1} minutes)", 
                 hashrate_mhs, expected_seconds, expected_minutes);
    }
    
    // Calculate how many hashes we did
    let hashes_done = 3_450_000_000.0;
    let probability_at_least_one = 1.0 - (1.0 - prob).powf(hashes_done);
    
    println!("\nWith {} billion hashes:", hashes_done / 1e9);
    println!("  Probability of finding at least one share: {:.2}%", 
             probability_at_least_one * 100.0);
}
