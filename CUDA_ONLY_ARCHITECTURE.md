# CUDA-Only Mining Architecture

## Project Statement

**rust-miner is a CUDA-ONLY cryptocurrency mining application.**

- ✅ **Requires:** NVIDIA GPU with CUDA support
- ❌ **No CPU mining:** Cryptocurrency mining only happens on NVIDIA GPUs
- ❌ **No CPU fallback:** If no CUDA GPU is available, mining cannot occur
- ❌ **No OpenCL:** Only CUDA backend is implemented
- ❌ **No alternative backends:** CUDA is the sole supported GPU interface

---

## Architecture Decision

### Why CUDA-Only?

1. **Performance Focus**
   - Direct access to NVIDIA GPU capabilities
   - Maximum kernel optimization potential
   - No abstraction overhead from multi-backend support

2. **Simplicity**
   - Single codebase path (no OpenCL, no HIP, no CPU variants)
   - Clear error messages when GPU not available
   - Easier maintenance and optimization

3. **Target Hardware**
   - Designed for NVIDIA GPUs (GTX 1050 Ti and better)
   - Modern CUDA Toolkit (12.0+)
   - Compute Capability 3.5+ required

### What CPU Does

The CPU is used ONLY for:
- **Network I/O:** Stratum pool protocol communication
- **Coordination:** Job distribution to GPU
- **Validation:** Share verification before submission
- **Monitoring:** Statistics and performance tracking

The CPU does **NOT**:
- Mine cryptocurrencies
- Compute hashes
- Perform PoW calculations
- Provide fallback mining

---

## GPU Detection and Initialization

### On Startup

```rust
// If no CUDA GPU found → Exit with clear error
if gpu_list.is_empty() {
    eprintln!("❌ No CUDA-compatible NVIDIA GPU found!");
    eprintln!("   This application requires an NVIDIA GPU.");
    eprintln!("   Please ensure:");
    eprintln!("   1. NVIDIA GPU is present in system");
    eprintln!("   2. NVIDIA drivers are installed");
    eprintln!("   3. CUDA Toolkit is installed");
    std::process::exit(1);
}
```

### GPU Requirements

| Requirement | Value |
|-------------|-------|
| GPU Type | NVIDIA only |
| Compute Capability | 3.5+ (Kepler) |
| VRAM | 2GB minimum |
| Driver | 450.0+ |
| CUDA Toolkit | 12.0+ |

### Supported GPUs

- ✅ GTX 1050 Ti and newer
- ✅ GTX 1660 SUPER (tested)
- ✅ RTX 2060 and newer
- ✅ RTX 3000 series and newer
- ✅ A100 and other enterprise GPUs

---

## Error Handling

### No GPU Available

```
Error: No CUDA-compatible GPU detected

This is a CUDA-only miner. GPU mining is required.

Required:
  • NVIDIA GPU (GeForce GTX 1050 Ti or better)
  • NVIDIA Drivers (450.0+)
  • CUDA Toolkit (12.0+)

Steps to fix:
  1. Verify GPU is installed: lspci | grep -i nvidia
  2. Check drivers: nvidia-smi
  3. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads

Exiting.
```

---

## CPU Usage Profile

While GPU performs mining, CPU is used for:

### Per-Second Operations
- 1 pool connection (persistent)
- 1 job check (lightweight)
- Share validation (if found)
- Statistics update

### CPU Load
- **Typical:** 5-10%
- **Peak:** 10-15% (during share submission)
- **Idle:** 1-3% (minimal driver polling)

### CPU Cores Used
- **Main thread:** Event loop
- **Worker threads:** 1-2 for network I/O
- **GPU driver:** Asynchronous CUDA operations

**No CPU cores are used for mining calculations.**

---

## Performance Implications

### Benefits of CUDA-Only
- No overhead from backend abstraction
- Direct kernel optimization
- Predictable, high performance
- Clear failure modes

### Trade-offs
- Requires NVIDIA hardware (not AMD or Intel Arc)
- No portability to CPU-only systems
- Depends on NVIDIA CUDA ecosystem

---

## Future Extensibility

The architecture is designed for adding new **algorithms** (SHA256, Blake3, etc.) but NOT new backends.

### Possible Future Additions
```rust
// New algorithms on same CUDA backend
pub enum Algorithm {
    QHash,      // ✅ Implemented
    SHA256,     // ❌ Future
    Blake3,     // ❌ Future
}

// Still CUDA-only - no CPU or OpenCL
pub trait CudaMiningAlgorithm {
    fn mine_cuda(&self, job: &Job) -> Result<Mining Result>;
}
```

### Will NOT Add
- ❌ CPU mining backend
- ❌ OpenCL backend
- ❌ HIP backend
- ❌ CPU fallback mining

---

## Summary

**rust-miner CUDA-only guarantees:**

✅ High-performance GPU mining on NVIDIA hardware
✅ Clean, focused codebase
✅ Clear error messages when requirements aren't met
✅ Predictable, optimizable architecture

❌ No CPU mining
❌ No CPU fallback
❌ No multi-backend complexity

---

**This is intentional by design.**

If you need CPU mining support, consider:
- [cpuminer-opt](https://github.com/JayDDee/cpuminer-opt)
- [xmrig](https://github.com/xmrig/xmrig)
- [lolminer](https://github.com/Lolliedieb/lolMiner-unofficial)

---

Last updated: November 20, 2025
