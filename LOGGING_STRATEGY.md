# Logging Strategy for rust-miner

## Overview
This document defines the logging levels and best practices used throughout the rust-miner project.

## Logging Levels

### `trace!` - Very Detailed Development Info
Used for **per-iteration** or **high-frequency** events that are too verbose for normal operation:
- Per-second hashrate samples (GPU #{} sampled hashrate)
- Individual hash computations and kernel timing
- Low-level backend details and device specs
- Sample-by-sample statistics

**Usage Pattern:**
```rust
tracing::trace!("GPU #{} sampled hashrate: {} hashes in {:.3}s = {:.2} MH/s",
    gpu_index, hashes_delta, elapsed, hashrate / 1_000_000.0);
```

**Enable with:** `RUST_LOG=trace`

---

### `debug!` - Development Information
Used for **non-critical but useful** information during development:
- Connection attempts (not just success/failure)
- Configuration parsing details
- Intermediate processing steps
- Performance observations

**Currently Not Heavily Used** - Reserved for future detailed debugging

---

### `info!` - Important State Changes
Used for **significant events** that should be logged in production:
- GPU detection and initialization
- Pool connection establishment
- Worker authorization and subscription
- Difficulty changes
- Share acceptance/rejection
- Mining task startup/shutdown
- Pool protocol events

**Usage Pattern:**
```rust
tracing::info!("GPU #{} mining task started with nonce offset: {}", gpu_index, current_nonce);
tracing::info!("Connecting to pool: {}", pool_url);
tracing::info!("Share accepted by pool");
```

**Enable with:** `RUST_LOG=info` (default recommended for production)

---

### `warn!` - Unexpected But Recoverable
Used when something **unexpected happens but operation continues:**
- Share rejected by pool (with reason)
- Device initialization warnings
- Connection retries
- Resource constraints

**Usage Pattern:**
```rust
tracing::warn!("Share rejected by pool: result={:?} error={:?}", response.result, response.error);
tracing::warn!("Failed to initialize CUDA device {}: {:?}", device_id, error);
```

---

### `error!` - Problems Requiring Attention
Used for **actual errors** that may impact operation:
- Connection failures and drops
- Mining failures
- Protocol errors
- Critical initialization failures

**Usage Pattern:**
```rust
tracing::error!("Connection closed by server");
tracing::error!("Failed to parse job: {}", error);
```

---

## Recommended Configuration

### Development
```bash
# Detailed tracing for active development
RUST_LOG=trace cargo run

# Normal debug with important events
RUST_LOG=debug cargo run
```

### Production (Recommended)
```bash
# Show all important events and problems
RUST_LOG=info cargo run

# Or simply (info is a reasonable default)
cargo run --release
```

### Troubleshooting
```bash
# Focus on specific modules
RUST_LOG=rust_miner::stratum=debug,rust_miner::cuda=debug cargo run

# Silence specific noisy modules
RUST_LOG=info,rust_miner::gpu=warn cargo run
```

---

## Current Logging Audit

### ✅ Correctly Categorized

**trace!** (high-frequency, per-iteration):
- GPU hashrate sampling every 1s
- Device memory/compute specs display

**info!** (important state changes):
- GPU detection and mining task startup
- Pool connection and authorization
- Difficulty updates
- Share submission status
- CUDA kernel compilation and initialization

**warn!** (unexpected but recoverable):
- Share rejections with error details
- Device initialization warnings

**error!** (actual problems):
- Connection drops
- Protocol parsing failures
- Message sending failures

---

## Best Practices

1. **High-Frequency Events** → Use `trace!`
   - Only enabled with `RUST_LOG=trace`
   - No performance impact when disabled
   - Safe to leave in hot paths

2. **State Changes** → Use `info!`
   - Pool events (connect, auth, difficulty)
   - GPU events (detection, task start/stop)
   - Share events (accepted/rejected)

3. **Unexpected Issues** → Use `warn!`
   - Share rejections (user wants to know)
   - Device warnings (non-blocking problems)

4. **Errors** → Use `error!`
   - Only for problems affecting operation
   - Connection failures, parsing errors

---

## Configuration Files

The logging system uses environment variables (`RUST_LOG`). No configuration file needed.

### Setup for First Run
```bash
# See everything
RUST_LOG=info cargo run --release

# For production servers, consider
RUST_LOG=info,tokio=warn,hyper=warn cargo run --release
```

---

## Adding New Logs

When adding new logging statements:

1. **Ask yourself:** "Will I want to see this in production?"
   - Yes → `info!` or higher
   - Only during development → `trace!` or `debug!`

2. **Ask yourself:** "Does this happen once or many times?"
   - Once per operation → `info!`
   - Many times per second → `trace!`

3. **Ask yourself:** "Is something going wrong?"
   - Unexpected but recoverable → `warn!`
   - Actual failure → `error!`

---

## Files with Logging

- `src/main.rs` - Mining loop, GPU stats (trace + info)
- `src/stratum/client.rs` - Pool protocol, authentication (info + warn + error)
- `src/cuda/mod.rs` - CUDA initialization (info)
- `src/gpu/cuda.rs` - Device detection (warn)

---

Last updated: November 20, 2025
