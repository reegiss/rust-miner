/// Mining algorithm trait and implementations
pub mod qhash;

/// Trait for mining hash algorithms
pub trait _HashAlgorithm {
    /// Get the algorithm name
    fn name(&self) -> &str;
    
    /// Compute hash of an 80-byte block header
    /// Returns 32-byte hash
    fn hash(&self, header: &[u8; 80]) -> [u8; 32];
    
    /// Check if hash meets target difficulty
    /// Target is in big-endian format (higher bytes = more significant)
    fn meets_target(&self, hash: &[u8; 32], target: &[u8; 32]) -> bool {
        // Compare as big-endian (MSB first)
        for i in 0..32 {
            if hash[i] < target[i] {
                return true;  // hash < target
            } else if hash[i] > target[i] {
                return false; // hash > target
            }
            // Continue if equal
        }
        false // hash == target (not strictly less than)
    }
}
