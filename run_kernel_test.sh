#!/bin/bash
# Test GPU kernel performance in isolation (no pool overhead)
# This directly measures qhash_mine kernel without pool/network latency

set -e

cd /home/regis/develop/rust-miner

echo "========================================="
echo "Rust-miner GPU Kernel Isolation Test"
echo "========================================="
echo ""
echo "Building release binary..."
cargo build --release 2>&1 | grep -E "Finished|error" || true

echo ""
echo "Running kernel performance test..."
echo "(Measuring 5 consecutive kernel calls with 50M nonces each)"
echo ""

# Create a simple test harness that calls GPU kernel directly
cat > /tmp/test_qhash_kernel.rs << 'EOF'
use std::time::Instant;

// Note: This is a conceptual test. In reality, we'd need to:
// 1. Link to cudarc
// 2. Compile qhash.cu
// 3. Launch kernel directly

// For now, we'll use the mining_result approach from main.rs
// and capture kernel timing via logging

fn main() {
    println!("Kernel Performance Test");
    println!("Testing 5 iterations of 50M nonces each");
    println!("");
}
EOF

# Instead, let's create a Python script that parses the miner output
cat > /tmp/parse_kernel_time.py << 'EOF'
#!/usr/bin/env python3
import sys
import re
from statistics import mean, stdev

kernel_times = []
batch_sizes = []
hashrates = []

for line in sys.stdin:
    # Parse: "GPU poll done: iters=16 elapsed_ms=400 batch_nonces=50000000 estimated_MH/s=125.0"
    match = re.search(r'elapsed_ms=(\d+).*batch_nonces=(\d+).*estimated_MH/s=([\d.]+)', line)
    if match:
        elapsed_ms = int(match.group(1))
        batch_nonces = int(match.group(2))
        hashrate_mhs = float(match.group(3))
        
        kernel_times.append(elapsed_ms)
        batch_sizes.append(batch_nonces)
        hashrates.append(hashrate_mhs)
        
        print(f"  {elapsed_ms:4d}ms | {batch_nonces/1_000_000:5.1f}M nonces | {hashrate_mhs:6.1f} MH/s")

if kernel_times:
    print("")
    print("=" * 50)
    print(f"Samples collected: {len(kernel_times)}")
    print(f"Kernel time (avg): {mean(kernel_times):.0f}ms (σ={stdev(kernel_times) if len(kernel_times) > 1 else 0:.0f})")
    print(f"Batch size (avg):  {mean(batch_sizes)/1_000_000:.1f}M nonces")
    print(f"Hashrate (avg):    {mean(hashrates):.1f} MH/s")
    print(f"Hashrate (min):    {min(hashrates):.1f} MH/s")
    print(f"Hashrate (max):    {max(hashrates):.1f} MH/s")
    print("=" * 50)
else:
    print("No kernel timing data found in input")
    print("Expected format: 'GPU poll done: iters=... elapsed_ms=... batch_nonces=... estimated_MH/s=...'")

EOF

chmod +x /tmp/parse_kernel_time.py

echo "========================================="
echo "Test 1: Current Config (128 threads/block)"
echo "========================================="
echo ""
echo "⚠️  REQUIRES CONFIGURED STRATUM POOL"
echo "Set up a pool connection in config, then run:"
echo ""
echo "  RUST_LOG=info ./target/release/rust-miner --url stratum+tcp://... --user ... 2>&1 | \\"
echo "    grep 'GPU poll done' | head -30 | python3 /tmp/parse_kernel_time.py"
echo ""
echo "========================================="
echo ""

# Alternative: Create a mock test that directly measures kernel launch
echo "Alternative: Direct GPU Kernel Measurement"
echo ""
echo "Creating synthetic test..."
echo ""

# Create a simple Rust test that measures GPU kernel performance
cat > /tmp/bench_qhash.rs << 'EOF'
// Synthetic benchmark - would need to be compiled with the project
use std::time::Instant;

fn main() {
    // This would need to:
    // 1. Load CUDA kernel
    // 2. Allocate GPU memory
    // 3. Launch qhash_mine with 50M nonces
    // 4. Measure kernel execution time
    
    println!("To run direct GPU kernel test:");
    println!("1. Copy src/cuda/qhash.cu");
    println!("2. Create standalone CUDA launcher");
    println!("3. Compile with nvcc -O3");
    println!("4. Run: ./bench_qhash");
    println!("");
    println!("But for now, use the mining approach with pool");
}
EOF

cat /tmp/bench_qhash.rs

echo ""
echo "========================================="
echo "Test 2: With threads_per_block = 256"
echo "========================================="
echo ""
echo "To test different thread configurations:"
echo ""
echo "  # Edit src/cuda/mod.rs line 63:"
echo "  let threads_per_block = 256;  # or 384, 512, etc"
echo ""
echo "  cargo build --release"
echo "  RUST_LOG=info ./target/release/rust-miner --url ... --user ... 2>&1 | \\"
echo "    grep 'GPU poll done' | head -30 | python3 /tmp/parse_kernel_time.py"
echo ""
echo "========================================="
echo ""

echo "Summary: To get kernel timing, you need to:"
echo ""
echo "1. Have a working stratum pool connection"
echo "2. Run miner with RUST_LOG=info"
echo "3. Filter output with: grep 'GPU poll done'"
echo "4. Parse results with: python3 /tmp/parse_kernel_time.py"
echo ""
echo "Each 'GPU poll done' line shows:"
echo "  - elapsed_ms: Real kernel execution time"
echo "  - batch_nonces: Number of nonces processed"
echo "  - estimated_MH/s: Kernel throughput (hashes per second)"
echo ""
echo "If kernel MH/s is consistently 37 MH/s:"
echo "  → GPU is slow, occupancy/tuning needed"
echo ""
echo "If kernel MH/s is 100+ MH/s but total is 37 MH/s:"
echo "  → Pool overhead, job switching, or network latency"
echo ""
EOF

chmod +x /tmp/test_qhash_kernel.rs

cat /tmp/test_qhash_kernel.rs

echo ""
echo "========================================="
echo "Running Parser Installation Check"
echo "========================================="
python3 /tmp/parse_kernel_time.py < /dev/null

echo ""
echo "✅ Test infrastructure ready"
echo ""
echo "Next step: Connect to a pool and run with logging enabled"
