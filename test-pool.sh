#!/bin/bash
# Quick pool test script for rust-miner

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Rust Miner - Pool Test ===${NC}\n"

# Check if wallet address provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Wallet address required${NC}"
    echo ""
    echo "Usage: ./test-pool.sh YOUR_WALLET_ADDRESS [mode]"
    echo ""
    echo "Modes:"
    echo "  dev     - Development build with info logs (default)"
    echo "  debug   - Development build with debug logs"
    echo "  release - Optimized release build"
    echo ""
    echo "Example:"
    echo "  ./test-pool.sh Qxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    echo "  ./test-pool.sh Qxxxxxxxxxxxxxxxxxxxxxxxxxxxx debug"
    echo "  ./test-pool.sh Qxxxxxxxxxxxxxxxxxxxxxxxxxxxx release"
    exit 1
fi

WALLET="$1"
MODE="${2:-dev}"

# Pool configuration
POOL_URL="qubitcoin.luckypool.io:8610"
ALGO="qhash"
PASS="x"

echo -e "${GREEN}Configuration:${NC}"
echo "  Pool:   $POOL_URL"
echo "  Algo:   $ALGO"
echo "  Wallet: $WALLET"
echo "  Mode:   $MODE"
echo ""

case "$MODE" in
    dev)
        echo -e "${YELLOW}Building in dev mode...${NC}"
        cargo build
        echo ""
        echo -e "${GREEN}Starting miner (info logs)...${NC}"
        RUST_LOG=info cargo run -- \
            --algo "$ALGO" \
            --url "$POOL_URL" \
            --user "$WALLET" \
            --pass "$PASS"
        ;;
    
    debug)
        echo -e "${YELLOW}Building in dev mode...${NC}"
        cargo build
        echo ""
        echo -e "${GREEN}Starting miner (debug logs)...${NC}"
        RUST_LOG=debug cargo run -- \
            --algo "$ALGO" \
            --url "$POOL_URL" \
            --user "$WALLET" \
            --pass "$PASS"
        ;;
    
    release)
        echo -e "${YELLOW}Building in release mode (optimized)...${NC}"
        cargo build --release
        echo ""
        echo -e "${GREEN}Starting miner (release build)...${NC}"
        RUST_LOG=info ./target/release/rust-miner \
            --algo "$ALGO" \
            --url "$POOL_URL" \
            --user "$WALLET" \
            --pass "$PASS"
        ;;
    
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Valid modes: dev, debug, release"
        exit 1
        ;;
esac
