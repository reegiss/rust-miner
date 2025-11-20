# Performance Analysis & Optimization Guide

## Overview
This document outlines performance benchmarking methodology, profiling tools, and optimization opportunities for rust-miner running on NVIDIA GTX 1660 SUPER.

---

## 1. Baseline Performance Metrics

### Current Performance (GTX 1660 SUPER)
- **Hashrate (Analytical QHash):** ~295 MH/s (stable)
- **CPU Usage:** ~6% (efficient polling, non-blocking)
- **GPU Memory:** ~512 MB (CUDA kernel + lookup tables)
- **Power Draw:** ~80W (estimated, GPU-only mining)
- **Share Submission Rate:** ~3-5 shares/minute (varies with difficulty)

### Hardware Specifications
```
GPU: NVIDIA GeForce GTX 1660 SUPER
  - VRAM: 4GB GDDR6
  - Compute Capability: 7.5
  - CUDA Cores: 1408
  - Boost Clock: 1815 MHz
  - Memory Bandwidth: 336 GB/s
  - Thermal Design Power (TDP): 125W
```

---

## 2. Benchmarking Methodology

### 2.1 Hashrate Benchmarks

#### Test Setup
```bash
# Production logging (minimal overhead)
RUST_LOG=info cargo build --release
./target/release/rust-miner \
  --algo qhash \
  --url qubitcoin.luckypool.io:8610 \
  --user <wallet>.<worker> \
  --pass x \
  --duration 3600  # Run for 1 hour

# Development logging (trace overhead measurement)
RUST_LOG=trace cargo build --release
./target/release/rust-miner \
  --algo qhash \
  --url qubitcoin.luckypool.io:8610 \
  --user <wallet>.<worker> \
  --pass x \
  --duration 3600
```

#### Metrics to Capture
```
Per 10 minutes:
  - Hashrate (10s window moving average)
  - Hashrate (60s window moving average)
  - Kernel execution time (ms)
  - Hashes per kernel launch
  - Share submissions
  - Share accepts
  - Pool latency (ms)

Final Stats:
  - Total hashes computed
  - Total shares found
  - Share acceptance rate (%)
  - Uptime
  - Average CPU usage
  - GPU temperature (if available)
```

### 2.2 Logging Overhead Comparison

#### Test Matrix
```
RUST_LOG Settings to Compare:
  1. off                              (no logging)
  2. info                             (production default)
  3. debug                            (development)
  4. trace                            (maximum verbosity)
  5. info,rust_miner::stratum=debug  (pool debugging)
```

#### Measurement Points
```bash
# Measure time for 100 iterations
time ./target/release/rust-miner --algo qhash --duration 60 --iterations 100

# CPU profiling during benchmark
perf stat -e cycles,instructions,cache-references,cache-misses \
  ./target/release/rust-miner --algo qhash --duration 60

# Memory profiling
valgrind --tool=massif ./target/release/rust-miner --algo qhash --duration 60
```

---

## 3. Profiling Tools & Techniques

### 3.1 Linux Profiling (Recommended)

#### Using `perf`
```bash
# Record performance data
perf record -g --call-graph=dwarf -F 99 ./target/release/rust-miner --duration 60

# View results
perf report

# Flame graph generation
perf record -F 99 -g ./target/release/rust-miner --duration 60
perf script > out.perf
git clone https://github.com/brendangregg/FlameGraph
FlameGraph/stackcollapse-perf.pl out.perf > out.folded
FlameGraph/flamegraph.pl out.folded > out.svg
# Open out.svg in browser
```

#### Using `flamegraph` crate
```bash
# Install flamegraph
cargo install flamegraph

# Generate flamegraph
cargo flamegraph --bin rust-miner -- --algo qhash --duration 60

# View
open flamegraph.svg  # macOS
firefox flamegraph.svg  # Linux
```

#### CPU Cycles Analysis
```bash
# Count CPU events
perf stat -e cycles,instructions,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses \
  ./target/release/rust-miner --duration 60

# Identify hot functions
perf top -e cycles:u
```

### 3.2 Memory Profiling

#### Using `valgrind`
```bash
# Massif - heap memory profiler
valgrind --tool=massif --massif-out-file=massif.out \
  ./target/release/rust-miner --algo qhash --duration 60

# View results
ms_print massif.out

# Focus on allocations
valgrind --tool=massif --ignore-fn='std::*' \
  ./target/release/rust-miner --algo qhash --duration 60
```

#### Heap Dump
```bash
# Find memory leaks
valgrind --leak-check=full --show-leak-kinds=all \
  ./target/release/rust-miner --algo qhash --duration 60
```

### 3.3 GPU Profiling (NVIDIA Tools)

#### Using `nvidia-smi`
```bash
# Monitor GPU in real-time
nvidia-smi -l 1  # Refresh every second

# Query specific metrics
nvidia-smi --query-gpu=index,name,clock_current,memory.used,power.draw,temperature.gpu \
  --format=csv -l 1
```

#### Using NVIDIA Profiler
```bash
# Requires CUDA Toolkit
nsys profile -o results ./target/release/rust-miner --algo qhash --duration 60

# Analyze results
nsys stats results.nsys-rep
```

---

## 4. Key Performance Indicators (KPIs)

### 4.1 Mining Metrics

| Metric | Target | Current | Unit |
|--------|--------|---------|------|
| Hashrate (10s window) | > 290 | 295 | MH/s |
| Hashrate (60s window) | > 290 | 295 | MH/s |
| Hashrate (15m window) | > 285 | 295 | MH/s |
| Share submission rate | > 2/min | 3-5 | shares/min |
| Share acceptance rate | > 95 | 98 | % |
| CPU usage | < 10 | 6 | % |
| GPU memory | < 1000 | 512 | MB |
| Kernel execution time | 700-900 | 850 | ms |

### 4.2 Stability Metrics

| Metric | Target | Unit |
|--------|--------|------|
| Uptime | > 24h | hours |
| Connection stability | > 99 | % |
| Share loss rate | < 1 | % |
| Pool disconnections | 0 | per hour |
| Stale share rate | < 5 | % |

### 4.3 Logging Overhead

| Log Level | Expected CPU Overhead | Message Rate |
|-----------|----------------------|--------------|
| off | 0% | - |
| info | 0-2% | ~10/sec |
| debug | 2-5% | ~50/sec |
| trace | 5-15% | ~500/sec |

---

## 5. Optimization Opportunities

### 5.1 High-Priority Optimizations

#### 1. Kernel Launch Optimization
```
Current:  ~850ms per kernel execution
Target:   < 700ms
Approach: Profile kernel, reduce register pressure, increase occupancy
```

#### 2. Memory Transfer Optimization
```
Current:  Host↔GPU transfers for each job
Target:   Batch multiple jobs, reduce transfers
Impact:   Reduce PCIe latency, improve throughput
```

#### 3. Nonce Distribution
```
Current:  Linear nonce distribution across iterations
Target:   Optimize for GPU occupancy based on compute capability
Impact:   Better GPU utilization, fewer idle threads
```

### 5.2 Medium-Priority Optimizations

#### 1. Pool Communication
```
Current:  Blocking I/O, potential latency spikes
Target:   Non-blocking connection pooling, pipelining
Impact:   Faster share submission, reduced stale rate
```

#### 2. Lock Contention
```
Current:  Multiple locks for GPU access and stats
Target:   Lock-free structures where possible
Impact:   Reduce context switching, improve responsiveness
```

#### 3. Logging Performance
```
Current:  Synchronous logging with tracing
Target:   Async logging, batched writes
Impact:   Reduce I/O blocking, lower CPU usage
```

### 5.3 Low-Priority (Future)

#### 1. Algorithm Optimizations
```
Target: Implement SHA256 as alternative PoW
Impact: Comparison baseline, feature parity with mainstream pools
```

#### 2. Multi-GPU Optimization
```
Target: Implement work-stealing queue for load balancing
Impact: Better scaling across heterogeneous GPU setups
```

---

## 6. Profiling Checklist

### Before Running Benchmarks
- [ ] System idle, no other GPU apps running
- [ ] GPU drivers up to date
- [ ] CUDA Toolkit matching `cudarc` version
- [ ] Power settings: max performance (not power-saving)
- [ ] Thermal paste fresh (if applicable)
- [ ] Room temperature stable (~22°C)

### During Benchmarks
- [ ] Monitor GPU temperature (target: < 70°C)
- [ ] Monitor power draw stability
- [ ] Capture system logs
- [ ] Record pool connectivity
- [ ] Note any thermal throttling

### After Benchmarks
- [ ] Compare against baseline
- [ ] Document variance (min/max/avg)
- [ ] Identify outliers
- [ ] Generate performance reports
- [ ] Archive raw profiling data

---

## 7. Performance Regression Detection

### Continuous Monitoring
```bash
# Run baseline every commit
cargo build --release
RUST_LOG=info time ./target/release/rust-miner --duration 300

# Compare hashrate against previous baseline
# Alert if > 5% regression
```

### Automated Benchmarks
```bash
# Add to CI pipeline
cargo bench --bench mining_bench --features cuda
```

---

## 8. Reporting Template

### Performance Report

```
Date: [YYYY-MM-DD]
Duration: [X hours]
RUST_LOG: [level]
GPU: GTX 1660 SUPER
Driver: [version]
CUDA: [version]

RESULTS:
  Hashrate (10s avg):    [X] MH/s
  Hashrate (60s avg):    [X] MH/s
  Hashrate (15m avg):    [X] MH/s
  CPU Usage (avg):       [X]%
  GPU Memory:            [X] MB
  Shares Found:          [X]
  Share Acceptance:      [X]%
  Pool Latency (avg):    [X] ms
  Uptime:                [X]h [X]m [X]s

ANALYSIS:
  - Observations
  - Anomalies detected
  - Variance analysis
  - Recommendations

REGRESSION CHECK:
  vs Baseline: [+/-X]%
  Status: [OK / WARNING / FAIL]
```

---

## 9. Next Steps

1. **Immediate:** Run baseline benchmarks with all RUST_LOG levels
2. **Week 1:** Generate flamegraphs and identify hot paths
3. **Week 2:** Profile memory usage and identify leaks
4. **Week 3:** Implement high-priority optimizations
5. **Week 4:** Re-benchmark and compare results

---

## Resources

- [Linux Perf Wiki](https://perf.wiki.kernel.org/)
- [Flame Graphs](http://www.brendangregg.com/flamegraphs.html)
- [Valgrind Manual](https://valgrind.org/docs/manual/)
- [NVIDIA Profiler Docs](https://docs.nvidia.com/cuda/profiler-users-guide/)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)

---

Last updated: November 20, 2025
