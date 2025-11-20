# QHash Bottleneck Analysis

## Current Baseline
- **Hashrate**: 37 MH/s
- **Kernel Time**: 400ms (from Phase 1)
- **Threads/Block**: 128 (optimized)
- **GPU**: NVIDIA GTX 1660 SUPER (CC 7.5)

## Target
- **Hashrate**: 500 MH/s (13.5x improvement)
- **Kernel Time**: ~30ms (theoretical at 500 MH/s with current batch size)

## Possible Gargalos

### 1. SHA256_transform (lines 76-110)
- **Operations per thread**: 64 rounds × 5 operations = ~320 ops
- **ILP Required**: High (dependent chain: t1 → t2 → state rotation)
- **Register Pressure**: ~20 registers
- **Instruction Throughput**: Rotations (ROTR) can be 1 instruction or 3+ depending on NVRTC optimization

```cuda
// Line 99-104: Dependent chain
uint32_t t1 = h + EP1(e) + CH(e, f, g) + K[i] + w[i];  // 6+ ops
uint32_t t2 = EP0(a) + MAJ(a, b, c);                   // 5+ ops
h = g; g = f; f = e; e = d + t1;                        // 1 op (critical: depends on t1)
d = c; c = b; b = a; a = t1 + t2;                       // 1 op
```

**Dependency Graph Issue**: Each SHA256 round depends on previous round's `e` and `a` values. This creates a ~15 cycle dependency chain that limits parallelism.

### 2. Quantum Simulation (lines 417-443)
- **Operations per thread**: 16 qubits × 2 layers × (2 lookups + 1 multiply) ≈ 96 ops
- **Register Pressure**: ~24 registers (expectations[16] array)
- **Memory Access**: All local (no shared/global memory)
- **Branch Divergence**: None - all threads do same work

**Assessment**: NOT bottleneck (all local, no memory stall)

### 3. Memory Access Patterns
- **Header Load**: 76 bytes × (threads_per_block/warp_size) = 128/32 = 4 warps, coalesced ✓
- **Target Load**: 32 bytes × 4 warps, coalesced ✓
- **Hash1 Output**: Register-only, no write ✓
- **Nibbles**: Register-only, no write ✓
- **Final Hash Write**: One atomicCAS per block at solution (rare)

**Assessment**: Memory EFFICIENT

### 4. Occupancy Analysis
- **Threads per block**: 128
- **Max threads per SM (CC 7.5)**: 1024
- **Blocks per SM**: 1024/128 = 8 blocks
- **Occupancy**: 128/1024 = 12.5% (VERY LOW)

**Assessment**: CRITICAL ISSUE - only 12.5% occupancy means GPU underutilized

## Root Cause: SHA256_transform Dependency Chain

The SHA256 compression function has **inherent sequential dependency**:

```
Round i:
  t1 = h + EP1(e) + CH(e,f,g) + K[i] + w[i]    [depends on round i-1's e]
  t2 = EP0(a) + MAJ(a,b,c)                      [depends on round i-1's a]
  e = d + t1                                     [CRITICAL: depends on t1]
```

Each round depends on previous round's output. **Cannot unroll past ~2-3 iterations without extreme register pressure.**

With `#pragma unroll 64`, NVRTC expands all 64 iterations, but:
1. Each thread waits for previous round to complete
2. Limited ILP from other instructions
3. GPU has warp divergence if any thread takes different path

## Calculations

### Theoretical Maximum (unrealistic)
- GTX 1660 SUPER: 1408 CUDA cores
- Peak throughput: 1408 × 1536 MHz = ~2.16 TFLOPS
- SHA256 hash: ~300 cycles (conservative estimate with dependency chain)
- **Theoretical max**: (1408 × 1536 MHz) / 300 cycles = ~7.2 GH/s

### Realistic with Current Architecture
- Occupancy: 12.5% (huge waste)
- Dependency chain: ~15 cycles per round × 64 rounds = ~960 cycles per hash
- With memory latency: ~1200 cycles per hash
- Hashrate: (1408 cores × 1536 MHz / 1200) / 1M = **1.8 MH/s per SM**
- With 40 SMs: ~72 MH/s theoretical max for current config

**Our 37 MH/s = 51% of theoretical max** → Not terrible, but room for improvement

## Optimization Paths

### Path A: Increase Occupancy
- **Option 1**: Reduce threads/block → 96 or 64
  - Pro: More blocks/SM, better occupancy
  - Con: SHA256 still has same dependency chain
  - Expected improvement: ~10-20%

- **Option 2**: Increase threads/block → 256 or 512
  - Pro: Better utilize GPU compute if SHA256 not bottleneck
  - Con: More register pressure, may spill to local memory
  - Risk: Occupancy could drop if registers increase

### Path B: Reduce SHA256 Cycles
- **Option 1**: PTX inline assembly
  - Replace `#pragma unroll 64` with fully unrolled PTX
  - Target: 2-3x faster SHA256 via better scheduling
  - Risk: PTX assembly is hard to write and maintain
  - Realistic gain: 15-30% (not 13.5x)

- **Option 2**: Use optimized SHA256 kernel from libgcrypt or similar
  - External dependency
  - May not compile with NVRTC
  - Realistic gain: 20-40%

### Path C: Algorithm Redesign
- **Problem**: QHash itself may be fundamentally slow
- **Option**: Change block size, batching strategy, or computational kernel
- **Risk**: Requires validation against blockchain spec

## Recommendation

**Before PTX Assembly**, test:

1. **Occupancy Boost**: Change threads_per_block to 256 and measure
   ```bash
   # In mod.rs
   threads_per_block = 256;
   cargo build --release && ./target/release/rust-miner
   ```

2. **Current Gargalo Confirmation**: Use nsys/nvprof to profile
   ```bash
   nsys profile --stats=true ./target/release/rust-miner
   ```

3. **Validate Realistic Target**:
   - If 500 MH/s is achievable at all
   - Or if realistic target is ~100-150 MH/s

## Next Steps

1. **Measure occupancy** with current 128 threads/block
2. **Profile with nsys** to see where time is spent
3. **Test threads_per_block = 256** (or 192, 384, 512)
4. **If still bottleneck**: Only then pursue PTX assembly for SHA256

---

**Estimate for 500 MH/s**: 13.5x gain seems unrealistic for architectural tweaks.
- SHA256 dependency chain is fundamental
- Best realistic gain: 2-4x from occupancy + memory optimization
- To reach 500 MH/s would require: new GPU, new algorithm, or batch processing
