# Pool Testing Guide

## QubitCoin Pool Configuration

**Pool:** qubitcoin.luckypool.io:8610  
**Protocol:** Stratum V1  
**Algorithm:** qhash (quantum Proof-of-Work)

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

### Release Build (Optimized CPU Performance)
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

2. **Mining Phase:**
   - ✅ Receive mining.notify jobs
   - ✅ Calculate merkle root
   - ✅ Build block header (80 bytes)
   - ✅ Mine nonces 0-1,000,000
   - ✅ Calculate qhash for each nonce
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
- Normal for high difficulty
- CPU mining is slow (~100-500 H/s)
- May take hours/days to find share
- Consider GPU optimization

### Share Rejected
- Hash doesn't meet difficulty (rare, should be checked before submit)
- Nonce already found by another miner
- Job expired (new block found)
- Check ntime and difficulty calculation

## Performance Notes

### Current Performance (CPU)
- **Algorithm:** QHash (quantum PoW simulation)
- **Expected Rate:** ~100-500 H/s (depends on CPU)
- **Nonce Range:** 0-1,000,000 per job
- **Time per Range:** ~30s-300s (depends on CPU)

### Future Performance (GPU - CUDA)
- **Expected Rate:** ~100-500 kH/s (1000x speedup)
- **Recommendation:** Port to GPU for practical mining

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
*Replace YOUR_WALLET_ADDRESS with your actual QubitCoin wallet address*
