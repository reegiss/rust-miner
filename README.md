# rust-miner

A high-performance cryptocurrency mining application written in Rust with GPU acceleration support.

## ⚡ Features

- **CUDA Support** (Primary) - Maximum performance on NVIDIA GPUs
- **OpenCL Support** (Fallback) - Cross-platform GPU compatibility
- **CPU Mining** - Fallback for systems without GPU
- **Multi-threaded** - Efficient CPU mining with Rayon
- **High Performance** - Optimized for speed with zero-copy operations

## 🚀 Quick Start

```bash
# 1. Run automated setup
bash setup.sh

# 2. Build with CUDA (recommended for NVIDIA GPUs)
cargo build --release --features cuda

# 3. Run
./target/release/rust-miner
```

For detailed setup instructions, see [QUICKSTART.md](QUICKSTART.md) or [SETUP.md](SETUP.md).

## 🎯 Hardware Requirements

### Minimum
- CPU: Multi-core processor (4+ cores recommended)
- RAM: 4GB
- OS: Linux, macOS, or Windows

### Recommended
- CPU: 8+ cores (e.g., AMD Ryzen 5/7, Intel Core i5/i7)
- GPU: NVIDIA GTX 1060 or better (for CUDA)
- RAM: 8GB+
- OS: Linux (Ubuntu 20.04+)

## 🔧 Build Options

```bash
# CPU-only build
cargo build --release --features cpu-only

# CUDA build (NVIDIA GPUs - best performance)
cargo build --release --features cuda

# OpenCL build (AMD/Intel GPUs)
cargo build --release --features opencl

# All backends (auto-detect best available)
cargo build --release --features all-backends
```

## 📊 Performance

Approximate hash rates (varies by hardware):

| Hardware | Algorithm | Hash Rate |
|----------|-----------|-----------|
| GTX 1660 SUPER (CUDA) | SHA256 | ~600 MH/s |
| GTX 1660 SUPER (CUDA) | Ethash | ~26 MH/s |
| Ryzen 5 5600X (12 threads) | SHA256 | ~10 MH/s |
| Ryzen 5 5600X (12 threads) | Ethash | ~0.5 MH/s |

## 🏗️ Architecture

```
rust-miner/
├── src/
│   ├── main.rs           # Entry point
│   ├── mining/           # Mining engine
│   │   ├── engine.rs     # Core mining logic
│   │   ├── cuda.rs       # CUDA backend (primary)
│   │   ├── opencl.rs     # OpenCL backend (fallback)
│   │   └── cpu.rs        # CPU fallback
│   ├── blockchain/       # Blockchain interface
│   └── utils/            # Utilities and helpers
└── benches/              # Benchmarks
```

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
