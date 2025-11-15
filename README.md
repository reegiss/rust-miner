# rust-miner

A high-performance cryptocurrency mining application written in Rust with GPU acceleration support.

**Cross-Platform**: Runs on both Linux and Windows with identical features and performance.

## ⚡ Features

- **Cross-Platform** - Runs on Linux and Windows
- **CUDA Support** (Primary) - Maximum performance on NVIDIA GPUs
- **OpenCL Support** (Fallback) - AMD/Intel GPU compatibility
- **GPU Required** - No CPU fallback mining (GPU is mandatory)
- **Multi-threaded** - Efficient parallel processing with Rayon
- **High Performance** - Optimized for speed with zero-copy operations

## 🚀 Quick Start

### Linux
```bash
# 1. Run automated setup
bash setup.sh

# 2. Build with CUDA (recommended for NVIDIA GPUs)
cargo build --release --features cuda

# 3. Run
./target/release/rust-miner
```

### Windows
```powershell
# 1. Install Rust from https://rustup.rs/

# 2. Install CUDA Toolkit from NVIDIA website

# 3. Build with CUDA
cargo build --release --features cuda

# 4. Run
.\target\release\rust-miner.exe
```

For detailed setup instructions, see [QUICKSTART.md](QUICKSTART.md) or [SETUP.md](SETUP.md).

## 🎯 Hardware Requirements

### Minimum
- **GPU: REQUIRED** - NVIDIA GTX 1050 Ti or AMD RX 560 (minimum)
- CPU: Multi-core processor (4+ cores recommended)
- RAM: 4GB
- OS: Linux (Ubuntu 20.04+, Fedora 35+) or Windows 10/11 (64-bit)

### Recommended
- **GPU: NVIDIA GTX 1660 or better (CUDA preferred)**
- CPU: 8+ cores (e.g., AMD Ryzen 5/7, Intel Core i5/i7)
- RAM: 8GB+
- OS: Linux (Ubuntu 22.04+) or Windows 11

**⚠️ Important**: This application requires a GPU. Systems without a GPU cannot mine.

## 🔧 Build Options

```bash
# CUDA build (NVIDIA GPUs - best performance, recommended)
cargo build --release --features cuda

# OpenCL build (AMD/Intel GPUs)
cargo build --release --features opencl

# All backends (auto-detect best available GPU)
cargo build --release --features all-backends
```

**Note**: CPU-only builds are not supported. A GPU is required for mining.

## 📊 Performance

Approximate hash rates (varies by hardware):

| Hardware | Algorithm | Hash Rate |
|----------|-----------|-----------|
| GTX 1660 SUPER (CUDA) | SHA256 | ~600 MH/s |
| GTX 1660 SUPER (CUDA) | Ethash | ~26 MH/s |
| GTX 1050 Ti (CUDA) | SHA256 | ~250 MH/s |
| GTX 1050 Ti (CUDA) | Ethash | ~11 MH/s |
| RX 580 (OpenCL) | SHA256 | ~400 MH/s |
| RX 580 (OpenCL) | Ethash | ~20 MH/s |

**Note**: CPU mining is not supported. GPU is required.

## 🏗️ Architecture

```
rust-miner/
├── src/
│   ├── main.rs           # Entry point
│   ├── mining/           # Mining engine
│   │   ├── engine.rs     # Core mining logic
│   │   ├── cuda.rs       # CUDA backend (primary)
│   │   └── opencl.rs     # OpenCL backend (fallback)
│   ├── blockchain/       # Blockchain interface
│   └── utils/            # Utilities and helpers
└── benches/              # Benchmarks
```

**Note**: No CPU mining implementation. GPU backends only.

## 🧪 Testing

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run CUDA-specific tests
cargo test --features cuda

# Benchmarks
cargo bench --features cuda
```

## 📚 Documentation

- [**QUICKSTART.md**](QUICKSTART.md) - Get started quickly
- [**SETUP.md**](SETUP.md) - Detailed setup guide
- [**.github/copilot-instructions.md**](.github/copilot-instructions.md) - Development guidelines

## 🤝 Contributing

Contributions are welcome! Please read our development guidelines in `.github/copilot-instructions.md` for code conventions and best practices.

## 📄 License

[MIT License](LICENSE) - feel free to use this project for learning and development.

## ⚠️ Disclaimer

This software is for educational purposes. Please ensure compliance with local regulations regarding cryptocurrency mining.

## 🔗 Resources

- [Rust Book](https://doc.rust-lang.org/book/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [cudarc Documentation](https://docs.rs/cudarc)
- [OpenCL Specification](https://www.khronos.org/opencl/)

---

**Built with ❤️ using Rust**
