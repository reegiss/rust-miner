use sha2::{Sha256, Digest};

fn main() {
    // Test SHA-256 with a simple input that should produce a hash with leading zeros
    
    // Block header example (76 bytes without nonce)
    let mut header = vec![0u8; 80];
    
    // Set version
    header[0..4].copy_from_slice(&4u32.to_le_bytes());
    
    // Try different nonces to see if we can get a hash with leading zeros
    println!("Testing SHA-256 double hash (sha256d) for leading zeros:\n");
    
    let mut found_zeros = vec![];
    
    for nonce in 0u32..1000000u32 {
        // Set nonce in last 4 bytes
        header[76..80].copy_from_slice(&nonce.to_le_bytes());
        
        // Double SHA-256
        let hash1 = Sha256::digest(&header);
        let hash2 = Sha256::digest(&hash1);
        
        // Count leading zero bits
        let mut leading_zeros = 0;
        for &byte in hash2.as_slice() {
            if byte == 0 {
                leading_zeros += 8;
            } else {
                leading_zeros += byte.leading_zeros() as usize;
                break;
            }
        }
        
        if leading_zeros >= 16 {  // At least 2 zero bytes
            found_zeros.push((nonce, hex::encode(&hash2), leading_zeros));
            
            if found_zeros.len() >= 5 {
                break;
            }
        }
        
        if nonce % 100000 == 0 {
            print!(".");
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }
    }
    
    println!("\n\nFound {} hashes with >= 16 leading zero bits:", found_zeros.len());
    for (nonce, hash, zeros) in found_zeros {
        println!("  Nonce {:8}: {} ({} leading zero bits)", nonce, hash, zeros);
    }
    
    println!("\nNote: A difficulty 5 target requires approximately 35-36 leading zero bits");
    println!("      Target: 000000000000024c40... = ~35.5 leading zero bits");
}
