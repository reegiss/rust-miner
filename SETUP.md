# Development Environment - rust-miner

## ✅ Current Environment

### Hardware
- **CPU**: AMD Ryzen 5 5600X (6-Core, 12 Threads)
- **GPU**: NVIDIA GeForce GTX 1660 SUPER (6GB VRAM)
- **RAM**: 15GB available
- **OS**: Ubuntu 24.04 LTS

### Installed Software
- ✅ **GCC**: 13.3.0 (C/C++ compiler)
- ✅ **Git**: 2.43.0
- ✅ **NVIDIA Driver**: 580.95.05
- ✅ **CUDA**: 13.0

### Pending Software
- ❌ **Rust**: Not installed (required)
- ❌ **OpenCL**: Not installed (recommended for GPU)
- ❌ **CUDA Toolkit**: Verify complete installation

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

## 🎮 GPU Configuration

### ⭐ Recommendation: CUDA (NVIDIA) - PRIMARY PRIORITY

This project uses **CUDA as primary backend** for maximum performance on NVIDIA GPUs.
OpenCL is maintained only as fallback for compatibility with other hardware.

### Option 1: CUDA (NVIDIA) - **RECOMMENDED**
```bash
# Verify if CUDA toolkit is complete
nvcc --version

# If not installed, install CUDA Toolkit 13.0
# Ubuntu 24.04:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-13-0

# Add to PATH
echo 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version
nvidia-smi
```

### Option 2: OpenCL (Fallback - AMD/Intel or NVIDIA without CUDA)
```bash
# Install NVIDIA OpenCL runtime
sudo apt update
sudo apt install ocl-icd-libopencl1 opencl-headers clinfo

# Install NVIDIA support for OpenCL
sudo apt install nvidia-opencl-icd-580

# Verify installation
clinfo
```

### Option 3: Both - CUDA + OpenCL (Maximum Flexibility)
```bash
# Install CUDA (priority)
sudo apt install cuda-toolkit-13-0

# Add OpenCL as fallback
sudo apt install ocl-icd-libopencl1 opencl-headers clinfo nvidia-opencl-icd-580

# Test CUDA (primary)
nvcc --version
nvidia-smi

# Test OpenCL (fallback)
clinfo | grep "Device Name"
```

**🎯 Recommendation**: Use Option 3 (both) for development, but prioritize CUDA in production.

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

## 🚀 Initialize Project

```bash
# Create Rust project
cd /home/regis/develop/rust-miner
cargo init --name rust-miner

# Add .gitignore
cat > .gitignore << 'EOF'
/target/
Cargo.lock
**/*.rs.bk
*.pdb
.DS_Store
.vscode/
.idea/
*.swp
*.swo
*~
EOF

# Test initial build (CPU-only)
cargo build
cargo run

# Test build with CUDA (recommended)
cargo build --release --features cuda

# Or with all backends
cargo build --release --features all-backends

# Run tests
cargo test

# Check formatting
cargo fmt --check

# Run linter
cargo clippy
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

## 📊 System Benchmark

### CPU Test
```bash
# After installing Rust, create simple benchmark
cargo new --bin cpu-bench
cd cpu-bench

# Add test code and run
cargo build --release
time ./target/release/cpu-bench
```

### GPU Test
```bash
# After configuring OpenCL/CUDA, test with examples
# Will be implemented in main project
```

## ✅ Setup Checklist

- [ ] Install Rust via rustup
- [ ] Install components: clippy, rustfmt, rust-analyzer
- [ ] **Install CUDA Toolkit 13.0 (PRIORITY)**
- [ ] Install OpenCL runtime (optional fallback)
- [ ] Verify nvcc and nvidia-smi working
- [ ] Verify clinfo (if installed OpenCL)
- [ ] Install profiling tools (perf, valgrind)
- [ ] Install cargo-flamegraph
- [ ] Configure ~/.cargo/config.toml for optimization
- [ ] Install VSCode extensions
- [ ] Initialize project with cargo init
- [ ] Test CPU build: `cargo build --release`
- [ ] **Test CUDA build: `cargo build --release --features cuda`**
- [ ] Verify linter: cargo clippy

## 🎯 Next Steps

1. **Install Rust** (priority)
2. **Configure CUDA Toolkit 13.0** (primary backend - NVIDIA)
3. **Install OpenCL** (optional - fallback for other hardware)
4. **Initialize project structure** with cargo
5. **Configure dependencies** in Cargo.toml (cudarc as primary)
6. **Implement basic mining engine** (CPU first)
7. **Add CUDA support** (maximum priority)
8. **Add OpenCL** as fallback (optional)
9. **Benchmarking and optimization** (CUDA vs CPU)

---
**Available hardware**: Excellent for mining! 12 CPU threads + GTX 1660 SUPER (CUDA) is a solid configuration.
**Recommended backend**: CUDA for maximum performance on NVIDIA GPU.
