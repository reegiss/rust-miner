use rust_miner::mining::nbits_to_target;

fn main() {
    // Test case: nBits = 0x1a024c62 (difficulty 5)
    let nbits = 0x1a024c62;
    let target = nbits_to_target(nbits);
    
    println!("nBits: 0x{:08x}", nbits);
    println!("Target (hex): {}", hex::encode(&target));
    
    println!("\nTarget bytes:");
    for (i, &byte) in target.iter().enumerate() {
        if i % 8 == 0 {
            print!("\n{:02}: ", i);
        }
        print!("{:02x} ", byte);
    }
    println!("\n");
    
    // Calculate expected difficulty
    // difficulty = target_max / target_current
    // target_max = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
    
    println!("Expected target for difficulty 5:");
    println!("Should be approximately: target_max / 5");
    
    // For nBits 0x1a024c62:
    // Exponent = 0x1a = 26
    // Mantissa = 0x024c62
    // Target = mantissa * 256^(exponent - 3)
    //        = 0x024c62 * 256^23
    //        = 0x000000024c620000000000000000000000000000000000000000000000000000
    
    let exponent = (nbits >> 24) as usize;
    let mantissa = nbits & 0x00FFFFFF;
    
    println!("\nParsed values:");
    println!("Exponent: {} (0x{:02x})", exponent, exponent);
    println!("Mantissa: {} (0x{:06x})", mantissa, mantissa);
    println!("Position in target array: {}", 32 - exponent);
}
