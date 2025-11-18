# Pool Testing Guide

## QubitCoin Pool Configuration

**Pool:** qubitcoin.luckypool.io:8610  
**Protocol:** Stratum V1  
**Algorithm:** qhash (Quantum Proof-of-Work)

## Testing Commands

### Basic Test (Development Mode)
```bash
RUST_LOG=info cargo run -- \
  --algo qhash \
  --url qubitcoin.luckypool.io:8610 \
  --user YOUR_WALLET_ADDRESS \
  --pass x
```

### Debug Mode (Verbose Output)
```bash
RUST_LOG=debug cargo run -- \
  --algo qhash \
  --url qubitcoin.luckypool.io:8610 \
  --user YOUR_WALLET_ADDRESS \
  --pass x
```

### Release Build (Optimized, CUDA)
```bash
cargo build --release
RUST_LOG=info ./target/release/rust-miner \
  --algo qhash \
  --url qubitcoin.luckypool.io:8610 \
  --user YOUR_WALLET_ADDRESS \
  --pass x
```

## Expected Behavior

1. **Connection Phase:**
   - ✅ Connect to pool
   - ✅ Send mining.subscribe
   - ✅ Receive extranonce1 and extranonce2_size
   - ✅ Authenticate with mining.authorize

2. **Mining Phase (GPU-only):**
   - ✅ Receive mining.notify jobs
   - ✅ Calculate merkle root
   - ✅ Build block header (80 bytes)
   - ✅ Mine nonces in adaptive batches on GPU
   - ✅ qhash kernel computes final hash
   - ✅ Check if hash < target

3. **Share Submission (if found):**
   - 🎉 Display "SHARE FOUND!"
   - 📤 Submit to pool via mining.submit
   - ✅ Show "Share accepted!" or ❌ "Share rejected"
   - 📊 Update statistics

4. **Statistics Display:**
   - Total hashes calculated
   - Hash rate (H/s)
   - Shares: found/accepted/rejected

## Wallet Setup

You need a QubitCoin wallet address. Options:

1. **QubitCoin Core Wallet:**
   - Download from official QubitCoin repository
   - Generate new address
   - Format: Starts with 'Q' (e.g., Qxxxxxxxxxxxxxxxxxxxxxxxxxxxxx)

2. **Pool Account:**
   - Some pools allow username instead of wallet
   - Check pool documentation

## Troubleshooting

### Connection Issues
```
Error: Connection refused
```
- Check internet connection
- Verify pool URL and port
- Try alternative pool if available

### Authentication Failed
```
Error: Authorization failed
```
- Verify wallet address format
- Check pool accepts new miners
- Try different worker name (address.worker1)

### No Shares Found
- Normal for difficulty 5 on small runtimes
- Let it run longer or reduce difficulty on pool if available

### Share Rejected
- Hash doesn't meet difficulty (rare, should be checked before submit)
- Nonce already found by another miner
- Job expired (new block found)
- Check ntime and difficulty calculation

## Performance Notes

### Performance (GPU - CUDA)
- GTX 1660 SUPER: ~37 MH/s (indicative)
- RTX 3060: ~65 MH/s (indicative)

## Monitoring

Watch for these log messages:

- `✅ Connected and authenticated` - Pool connection successful
- `📋 New job received` - Mining work available
- `⛏️ Mining...` - Currently searching for share
- `🎉 SHARE FOUND!` - Valid share discovered
- `✅ Share accepted!` - Pool accepted share (earnings!)
- `❌ Share rejected` - Pool rejected (investigate why)

## Next Steps After Successful Test

1. ✅ Verify end-to-end functionality
2. 🔧 Increase nonce range (currently 1M may be too small)
3. 🚀 Port to GPU (CUDA kernel) for production use
4. 📊 Add real-time monitoring dashboard
5. 🔄 Implement continuous mining (loop until new job)

---
*Replace YOUR_WALLET_ADDRESS with your actual Qubitcoin wallet address*
