#!/bin/bash
# Quick benchmark test: occupancy vs hashrate trade-off
# Test different threads_per_block configurations

THREADS_PER_BLOCK_VALUES=(64 96 128 192 256 384 512)

echo "QHash Occupancy vs Hashrate Benchmark"
echo "======================================"
echo ""

for TPB in "${THREADS_PER_BLOCK_VALUES[@]}"; do
    echo "Testing threads_per_block = $TPB"
    
    # Calculate occupancy for GTX 1660 SUPER (CC 7.5)
    # Max threads per SM: 1024
    # Max blocks per SM: 8 (for CC 7.5 with 128 threads minimum)
    BLOCKS_PER_SM=$((1024 / TPB))
    if [ $BLOCKS_PER_SM -gt 8 ]; then
        BLOCKS_PER_SM=8
    fi
    OCCUPANCY=$((BLOCKS_PER_SM * TPB * 100 / 1024))
    
    echo "  Occupancy: $OCCUPANCY% (${BLOCKS_PER_SM} blocks × ${TPB} threads)"
    echo "  Estimated register usage: ~24 registers/thread"
    
    # Additional constraints for CC 7.5:
    # Max registers per SM: 65536
    # If more than 65536/TPB registers needed, occupancy drops
    REGS_AVAILABLE=$((65536 / TPB))
    if [ $REGS_AVAILABLE -lt 24 ]; then
        echo "  WARNING: Register pressure may reduce occupancy!"
    fi
    
    echo ""
done

echo "======================================"
echo ""
echo "Recommendation for immediate testing:"
echo "1. Measure current 128 threads/block baseline (should be ~37 MH/s)"
echo "2. Test 256 threads/block (should increase occupancy to 25%)"
echo "3. Test 512 threads/block (should reach max occupancy 50%)"
echo "4. Measure hashrate at each config"
echo ""
echo "If hashrate improves with higher threads_per_block:"
echo "  → Occupancy WAS bottleneck, not SHA256 itself"
echo "  → PTX assembly unlikely to help much"
echo ""
echo "If hashrate stays same or decreases:"
echo "  → SHA256 dependency chain IS bottleneck"
echo "  → PTX assembly MAY help 10-30%"
