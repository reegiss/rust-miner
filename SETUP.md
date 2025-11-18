# Setup Guide - rust-miner (CUDA-only)

## ✅ Current Environment

### Hardware
- **CPU**: AMD Ryzen 5 5600X (6-Core, 12 Threads)
- **GPU**: NVIDIA GeForce GTX 1660 SUPER (6GB VRAM)
- **RAM**: 15GB available
- **OS**: Ubuntu 24.04 LTS

### Installed Software (example)
- ✅ GCC / Clang
- ✅ Git
- ✅ NVIDIA Driver
- ✅ CUDA Toolkit (12.x or newer)

### Pending Software
- ❌ Rust (required)
- ❌ CUDA Toolkit (required)

## 📦 Rust Installation

### Recommended Method: rustup (via curl)
```bash
# Download and install rustup (Rust version manager)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# During installation, choose option 1 (default installation)
# After installation, reload environment
source "$HOME/.cargo/env"

# Verify installation
rustc --version
cargo --version
```

### Configure Rust for Performance
```bash
# Install stable toolchain (default)
rustup toolchain install stable

# Install nightly toolchain (for experimental features)
rustup toolchain install nightly

# Set stable as default
rustup default stable

# Add useful components
rustup component add clippy rustfmt rust-analyzer
```

## 🎮 GPU Configuration (CUDA-only)

This project requires an NVIDIA GPU with CUDA. OpenCL is not supported.

### Install CUDA Toolkit (Ubuntu example)
```bash
# Check
nvcc --version || true
nvidia-smi || true

# Ubuntu 24.04 example install (adjust to your distro)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-5

# Add to PATH (adjust version)
echo 'export PATH=/usr/local/cuda-12.5/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
nvidia-smi
```
> Any CUDA 12.x+ toolkit is fine. The project uses cudarc and JIT compiles the kernel.

## 🔧 Development Tools

### Performance Profiling
```bash
# perf (profiler Linux)
sudo apt install linux-tools-common linux-tools-generic linux-tools-$(uname -r)

# Allow profiling without sudo
echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid

# valgrind (memory profiler)
sudo apt install valgrind

# cargo-flamegraph (performance visualization)
cargo install flamegraph
```

### Benchmarking
```bash
# criterion (benchmark framework for Rust)
# Will be added in project's Cargo.toml
```

### Code Analysis
```bash
# cargo-watch (auto-rebuild)
cargo install cargo-watch

# cargo-expand (expand macros)
cargo install cargo-expand

# cargo-bloat (binary size analysis)
cargo install cargo-bloat
```

## 🚀 Build & Run

```bash
# Build (CUDA-only)
cargo build --release

# Run with your pool
./target/release/rust-miner \
	--algo qhash \
	--url qubitcoin.luckypool.io:8610 \
	--user YOUR_WALLET.WORKER \
	--pass x
```

## ⚡ Environment Optimization

### Optimized Build Configuration
Create/edit `~/.cargo/config.toml`:
```toml
[build]
# Use all available cores
jobs = 12

[target.x86_64-unknown-linux-gnu]
# Link-time optimization
rustflags = ["-C", "target-cpu=native"]

[profile.release]
# Maximum optimizations
opt-level = 3
lto = "fat"
codegen-units = 1
```

### Recommended VSCode Extensions
```bash
# rust-analyzer (language server)
code --install-extension rust-lang.rust-analyzer

# Better TOML
code --install-extension tamasfe.even-better-toml

# CodeLLDB (debugger)
code --install-extension vadimcn.vscode-lldb

# crates (dependency management)
code --install-extension serayuzgur.crates
```

## 📊 Sanity Checks
```bash
rustc --version
cargo --version
nvcc --version
nvidia-smi
```

## ✅ Setup Checklist

- [ ] Install Rust via rustup
- [ ] Install components: clippy, rustfmt, rust-analyzer
- [ ] Install CUDA Toolkit 12.x+ (PRIORITY)
- [ ] Verify nvcc and nvidia-smi working
- [ ] Verify clinfo (if installed OpenCL)
- [ ] Install profiling tools (perf, valgrind)
- [ ] Install cargo-flamegraph
- [ ] Configure ~/.cargo/config.toml for optimization
- [ ] Install VSCode extensions
- [ ] Initialize project with cargo init
- [ ] Test CPU build: `cargo build --release`
- [ ] **Test build: `cargo build --release`**
- [ ] Verify linter: cargo clippy

## 🎯 Notes

- Project is CUDA-only, GPU required. There is no CPU mining mode.
- Kernel is JIT-compiled via NVRTC at startup.
- Backend is initialized before connecting to the pool to fail-fast on invalid `--algo`.

---
**Available hardware**: Excellent for mining! 12 CPU threads + GTX 1660 SUPER (CUDA) is a solid configuration.
**Recommended backend**: CUDA for maximum performance on NVIDIA GPU.
