#!/bin/bash

# Instructions for [TAREFA 2] Real Pool Validation
# This script guides testing with a real Stratum pool

cat << 'EOF'

╔════════════════════════════════════════════════════════════════════════╗
║         [TAREFA 2] REAL POOL VALIDATION INSTRUCTIONS                 ║
║                  Testing Job Switching Fix                            ║
╚════════════════════════════════════════════════════════════════════════╝

📌 OBJECTIVE
═══════════════════════════════════════════════════════════════════════

Validate that job switching overhead fix improves hashrate from 37 MH/s 
to the expected 150-300 MH/s range when mining with a real Stratum pool.

═══════════════════════════════════════════════════════════════════════

📋 REQUIREMENTS
═══════════════════════════════════════════════════════════════════════

1. NVIDIA GPU with CUDA support
   • Current baseline: GTX 1660 SUPER (CC 7.5)
   • Other GPUs supported via CUDA Toolkit

2. CUDA Toolkit 12.0+
   • Verify: nvcc --version
   • Verify: nvidia-smi

3. Stratum Pool Access
   • Pool URL: <pool_address>:<port>
   • Wallet address: <your_wallet>
   • Password: (usually 'x' for QHASH pools)

4. Project compiled in release mode
   • Run: cargo build --release

═══════════════════════════════════════════════════════════════════════

🚀 QUICK START (Template)
═══════════════════════════════════════════════════════════════════════

STEP 1: Determine pool credentials
  Pool Address:    qhash.pool.example.com
  Pool Port:       3333
  Wallet:          your_wallet_address
  Password:        x

STEP 2: Run the miner
  
  cargo run --release -- \
    --algo qhash \
    --url stratumv1://qhash.pool.example.com:3333 \
    --user your_wallet_address \
    --pass x

STEP 3: Monitor output

  Look for lines like:
    "GPU: 180.50 MH/s | last_kernel=168ms"
    
  And periodic stats:
    "═══════════════════════════════════════════════════════════════"
    "GPU Status  | Hashrate: X.XX MH/s | Shares: N"
    "═══════════════════════════════════════════════════════════════"

STEP 4: Let it run for 10+ minutes

  Collect data for at least 600 seconds to get reliable average hashrate.
  
═══════════════════════════════════════════════════════════════════════

📊 SUCCESS CRITERIA
═══════════════════════════════════════════════════════════════════════

BASELINE (Before Fix):
  • Observed: 37 MH/s
  • Problem: Job switching checks during batch

AFTER FIX (Expected):
  • Lower bound: 150 MH/s (4x improvement)
  • Target: 200-250 MH/s (5-6x improvement)
  • Upper bound: 300 MH/s (8x improvement)

ACCEPTANCE THRESHOLD:
  ✅ PASS if hashrate >= 150 MH/s sustained
  ⚠️  WARN if hashrate 75-150 MH/s (partial improvement)
  ❌ FAIL if hashrate < 75 MH/s (no improvement or regression)

═══════════════════════════════════════════════════════════════════════

📈 WHAT TO MEASURE
═══════════════════════════════════════════════════════════════════════

1. Sustained Hashrate (Primary)
   • Look for "GPU: X.XX MH/s" in logs
   • Calculate average over 10-minute window
   • Expected: 150-300 MH/s

2. Kernel Latency (Secondary)
   • Look for "last_kernel=XXXms" in logs
   • Expected: 150-250ms (similar to 50M nonce batch)
   • If >> 300ms, check for pool communication delays

3. Warp Status (Diagnostic)
   • Look for work items queued per GPU
   • Expected: 50M nonces per batch
   • If < 10M, job switching may still be interrupting

4. Share Rate (Tertiary)
   • Number of shares found over time
   • Expected: Varies by difficulty
   • Should increase proportionally with hashrate

═══════════════════════════════════════════════════════════════════════

🔍 TROUBLESHOOTING
═══════════════════════════════════════════════════════════════════════

Issue: Hashrate stuck at 37 MH/s or lower
  Cause: Job switching fix not working or pool interaction issue
  Action:
    1. Verify src/main.rs line 318: Job check is AFTER batch
    2. Check pool connectivity: can miner connect?
    3. Run isolated benchmark: ./target/release/examples/bench_qhash
    4. If benchmark = 325 MH/s, issue is pool-side

Issue: Hashrate 75-150 MH/s (partial improvement)
  Cause: Some job switching overhead remains
  Action:
    1. Increase batch size from 50M to 75M or 100M
    2. Check CPU load: is host CPU saturated?
    3. Profile with: cargo build --release && ./target/release/rust-miner 2>&1 | grep -E "GPU:|kernel"
    4. May need further tuning

Issue: Kernel crashes or GPU error
  Cause: CUDA compilation or memory issue
  Action:
    1. Rebuild: cargo clean && cargo build --release
    2. Check CUDA: nvidia-smi
    3. Test kernel isolation: ./target/release/examples/bench_qhash

═══════════════════════════════════════════════════════════════════════

📋 DATA COLLECTION TEMPLATE
═══════════════════════════════════════════════════════════════════════

When running pool test, collect:

Test Name: Pool Validation - Job Switching Fix
Date: [TODAY]
GPU: [GPU Model from nvidia-smi]
Pool: [Pool Address]

Time    | Hashrate (MH/s) | Kernel (ms) | Shares Found
--------|-----------------|-------------|---------------
00:00   | ?.?? | ???       | N
05:00   | ?.?? | ???       | N
10:00   | ?.?? | ???       | N

Average Hashrate: X.XX MH/s
Min Hashrate:     Y.YY MH/s
Max Hashrate:     Z.ZZ MH/s

✓ PASS / ⚠️ WARN / ❌ FAIL

═══════════════════════════════════════════════════════════════════════

💡 TIPS FOR ACCURATE TESTING
═══════════════════════════════════════════════════════════════════════

1. Stabilize before measuring
   • Run for 1-2 minutes to warm up pool connection
   • Then start taking measurements

2. Avoid other GPU loads
   • Close games, other CUDA applications
   • Disable screen savers and power management
   • Keep GPU in high-performance mode:
     nvidia-smi -pm 1
     nvidia-smi -lgc 2400  # Max GPU clock

3. Log to file for analysis
   cargo run --release -- [args] 2>&1 | tee pool_test_$(date +%s).log

4. Parse results with:
   grep "GPU:" pool_test_*.log | awk '{print $3}' | sort -n

═══════════════════════════════════════════════════════════════════════

🎯 EXPECTED LOG OUTPUT
═══════════════════════════════════════════════════════════════════════

Sample successful run:

  🟢 CUDA Backend Initialized
  ✅ GPU: NVIDIA GeForce GTX 1660 SUPER [0]

  [Connected to pool]

  GPU: 205.30 MH/s | last_kernel=167ms
  GPU: 210.45 MH/s | last_kernel=168ms
  GPU: 198.76 MH/s | last_kernel=170ms
  
  ═══════════════════════════════════════════════════════════════
  GPU Status      | Hashrate: 204.84 MH/s (avg)
                  | Shares: 3 valid
                  | Kernel: 168ms avg
  ═══════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════════

❓ QUESTIONS?
═══════════════════════════════════════════════════════════════════════

After pool test, you should know:
  • Did hashrate improve from 37 MH/s?
  • To what range? (150-300 MH/s expected)
  • Are there any stability issues?
  • Does kernel time match expectations?

Based on results, next steps:
  ✅ If >= 150 MH/s: Job switching fix CONFIRMED. Proceed to occupancy tuning.
  ⚠️ If 75-150 MH/s: Partial improvement. Investigate pool overhead or batch sizes.
  ❌ If < 75 MH/s: No improvement. Debug pool interaction or GPU configuration.

═══════════════════════════════════════════════════════════════════════

EOF
