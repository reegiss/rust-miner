#!/bin/bash

##############################################################################
#                    rust-miner Benchmark Script                            #
#                                                                            #
# Usage: ./benchmark.sh [--duration SECS] [--gpu GPU_ID] [--all-logs]     #
#                                                                            #
# Examples:                                                                  #
#   ./benchmark.sh                    # 1 hour benchmark with info logging  #
#   ./benchmark.sh --duration 600     # 10 minute benchmark                #
#   ./benchmark.sh --all-logs         # Test all logging levels            #
#                                                                            #
##############################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DURATION=3600  # 1 hour
GPU_ID=0
POOL_URL="qubitcoin.luckypool.io:8610"
WALLET="bc1qacadts4usj2tjljwdemfu44a2tq47hch33fc6f"
WORKER="benchmark-$(date +%s)"
TEST_ALL_LOGS=false
OUTPUT_DIR="./benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --all-logs)
            TEST_ALL_LOGS=true
            shift
            ;;
        --help)
            grep "^#" "$0" | tail -n +2
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           rust-miner Benchmark Suite                      ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to run a single benchmark
run_benchmark() {
    local log_level=$1
    local test_name=$2
    local output_file="${OUTPUT_DIR}/benchmark_${TIMESTAMP}_${log_level}.log"
    
    echo -e "${YELLOW}▶ Running: ${test_name}${NC}"
    echo "  Duration: $DURATION seconds"
    echo "  Log Level: $log_level"
    echo "  Output: $output_file"
    echo ""
    
    # Build with optimizations
    echo -e "${BLUE}  Building release binary...${NC}"
    cargo build --release 2>&1 | grep -E "(Compiling|Finished)" || true
    
    # Run benchmark
    echo -e "${BLUE}  Running benchmark...${NC}"
    
    RUST_LOG="$log_level" \
    /usr/bin/time -v \
    ./target/release/rust-miner \
        --algo qhash \
        --url "$POOL_URL" \
        --user "${WALLET}.${WORKER}" \
        --pass x \
        --gpu "$GPU_ID" \
        2>&1 | tee -a "$output_file"
    
    echo ""
    echo -e "${GREEN}✓ Benchmark complete: $output_file${NC}"
    echo ""
}

# Check prerequisites
echo -e "${YELLOW}▶ Checking prerequisites...${NC}"

if ! command -v cargo &> /dev/null; then
    echo -e "${RED}✗ Rust/Cargo not found${NC}"
    exit 1
fi

if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ NVIDIA GPU not found or nvidia-smi not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Cargo found${NC}"
echo -e "${GREEN}✓ NVIDIA GPU found${NC}"
echo ""

# Show GPU info
echo -e "${YELLOW}▶ GPU Information:${NC}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total \
    --format=csv,noheader | awk '{print "  GPU '$GPU_ID': " $2 " (Driver: " $3 ", VRAM: " $4 ")"}'
echo ""

# Show system info
echo -e "${YELLOW}▶ System Information:${NC}"
echo "  CPU: $(grep 'model name' /proc/cpuinfo | head -1 | awk -F: '{print $2}' | xargs)"
echo "  RAM: $(free -h | awk '/^Mem:/ {print $2}')"
echo "  Kernel: $(uname -r)"
echo ""

# Prompt to confirm
echo -e "${YELLOW}▶ Benchmark Configuration:${NC}"
echo "  Duration: $DURATION seconds"
echo "  Pool: $POOL_URL"
echo "  Worker: $WORKER"
echo "  Output Directory: $OUTPUT_DIR"
echo ""

read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Benchmark cancelled."
    exit 0
fi

echo ""

# Run benchmarks
if [ "$TEST_ALL_LOGS" = true ]; then
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}Testing all logging levels (may take several hours)${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    
    run_benchmark "info" "Production (info)"
    run_benchmark "debug" "Development (debug)"
    run_benchmark "trace" "Full Trace (trace)"
    run_benchmark "rust_miner::stratum=debug" "Pool Debugging (stratum=debug)"
    
else
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}Running Production Benchmark (RUST_LOG=info)${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    
    run_benchmark "info" "Production Benchmark"
fi

# Generate summary
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Benchmark Results Summary${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "Log files saved to: ${OUTPUT_DIR}/"
echo ""

# Try to extract key metrics
if [ -f "${OUTPUT_DIR}/benchmark_${TIMESTAMP}_info.log" ]; then
    echo -e "${YELLOW}▶ Key Metrics from Production Run:${NC}"
    
    # Extract from final statistics
    grep -E "(Hashrate|Total Hashes|Shares|Acceptance)" \
        "${OUTPUT_DIR}/benchmark_${TIMESTAMP}_info.log" | tail -10 || echo "  (Stats not available in log)"
    
    echo ""
fi

echo -e "${GREEN}Next steps:${NC}"
echo "  1. Review logs: cat ${OUTPUT_DIR}/benchmark_${TIMESTAMP}_*.log"
echo "  2. Compare results with baseline"
echo "  3. Check for regressions (target: > 290 MH/s)"
echo "  4. Analyze with flamegraph for profiling:"
echo "     perf record -g ./target/release/rust-miner --duration 300"
echo ""

echo -e "${GREEN}Benchmark completed!${NC}"
