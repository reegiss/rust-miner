# rust-miner

A high-performance cryptocurrency mining application written in Rust with GPU acceleration support.

**Cross-Platform**: Runs on both Linux and Windows with identical features and performance.

## ⚡ Features

- **Cross-Platform** - Runs on Linux and Windows
- **CUDA Support** - Optimized for NVIDIA GPUs (GPU required)
- **High Performance** - Zero-copy GPU operations, kernel returns hash directly
- **GPU Required** - No CPU fallback mining (dedicated GPU hardware mandatory)
- **Stratum V1 Protocol** - Compatible with standard mining pools
- **QHash Algorithm** - Quantum-resistant mining algorithm support
- **Adaptive Batch Sizing** - Dynamic nonce range optimization
- **Low CPU Usage** - Efficient non-blocking GPU polling (~6% CPU)

## 🚀 Quick Start

### Linux
```bash
# 1. Install dependencies
sudo apt update
sudo apt install -y build-essential curl git
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 2. Install CUDA Toolkit (NVIDIA GPU required)
# Download from: https://developer.nvidia.com/cuda-downloads

# 3. Clone and build
git clone https://github.com/yourusername/rust-miner.git
cd rust-miner
cargo build --release

# 4. Run with pool
./target/release/rust-miner --algo qhash --url pool.example.com:8610 --user YOUR_WALLET.WORKER --pass x
```

### Windows
```powershell
# 1. Install Rust from https://rustup.rs/

# 2. Install CUDA Toolkit from NVIDIA website

# 3. Clone and build
git clone https://github.com/yourusername/rust-miner.git
cd rust-miner
cargo build --release

# 4. Run with pool
.\target\release\rust-miner.exe --algo qhash --url pool.example.com:8610 --user YOUR_WALLET.WORKER --pass x
```

For detailed setup instructions, see [QUICKSTART.md](QUICKSTART.md) or [SETUP.md](SETUP.md).

## 🎯 Hardware Requirements

### Minimum
- **GPU: NVIDIA GTX 1050 Ti or better (CUDA required)**
- CPU: Multi-core processor (4+ cores recommended)
- RAM: 4GB
- OS: Linux (Ubuntu 20.04+) or Windows 10/11 (64-bit)
- CUDA Toolkit 12.0+

### Recommended
- **GPU: NVIDIA GTX 1660 SUPER or better**
- CPU: 8+ cores (for network and coordination tasks)
- RAM: 8GB+
- OS: Linux (Ubuntu 22.04+) or Windows 11
- CUDA Toolkit 12.0+

**⚠️ Important**: This application requires an NVIDIA GPU with CUDA support. Systems without compatible NVIDIA hardware cannot mine.

## 🔧 Build Options

```bash
# Standard CUDA build (NVIDIA GPUs)
cargo build --release

# Development build with debug symbols
cargo build

# Run tests
cargo test
```

**Note**: CUDA is the only supported backend. CPU mining is not available.

## 📊 Performance

Measured hash rates on real hardware:

| Hardware | Algorithm | Hash Rate | Power Usage |
|----------|-----------|-----------|-------------|
| **GTX 1660 SUPER (CUDA)** | **QHash** | **37.40 MH/s** | ~125W |
| GTX 1660 SUPER (CUDA) | SHA256d | ~600 MH/s | ~125W |
| GTX 1050 Ti (CUDA) | QHash | ~18 MH/s | ~75W |
| RTX 3060 (CUDA) | QHash | ~65 MH/s | ~170W |

**Performance Features**:
- ✅ GPU returns hash directly (no CPU recomputation)
- ✅ Adaptive batch sizing (targets 700-900ms per kernel)
- ✅ Non-blocking polling with ~6% CPU usage
- ✅ Zero-copy memory operations where possible

**Note**: CPU mining is not supported. CUDA-capable NVIDIA GPU is required.

## 🏗️ Architecture

```
rust-miner/
├── src/
│   ├── main.rs              # Entry point, mining orchestration
│   ├── cli.rs               # Command-line interface
│   ├── mining.rs            # Mining coordination layer
│   ├── cuda/
│   │   ├── mod.rs           # CUDA wrapper (Rust)
│   │   └── qhash.cu         # QHash kernel (CUDA C++)
│   ├── stratum/
│   │   ├── client.rs        # Stratum V1 client
│   │   └── protocol.rs      # Protocol types
│   ├── algorithms/
│   │   ├── mod.rs           # Algorithm trait
│   │   └── qhash.rs         # QHash CPU (testing only)
│   └── gpu/
│       ├── mod.rs           # GPU detection
│       └── cuda.rs          # CUDA device info
└── .github/
    └── copilot-instructions.md  # Development guidelines
```

**Architecture Principles**:
- GPU-mandatory design (no CPU fallback)
- CUDA-only backend (no OpenCL)
- Kernel returns (nonce, hash) directly - eliminates CPU recomputation
- Efficient spawn_blocking + adaptive sleep polling (~6% CPU)

## 🧪 Testing

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run CUDA-specific tests (requires GPU)
cargo test --test cuda_tests

# Check code without building
cargo check

# Format code
cargo fmt

# Run linter
cargo clippy -- -D warnings
```

## 📚 Documentation

- [**QUICKSTART.md**](QUICKSTART.md) - Get started quickly
- [**SETUP.md**](SETUP.md) - Detailed setup guide
- [**.github/copilot-instructions.md**](.github/copilot-instructions.md) - Development guidelines and architecture

## 🎯 Usage

```bash
# Basic usage
rust-miner --algo qhash --url pool.example.com:8610 --user WALLET.WORKER --pass x

# With specific GPU
rust-miner --algo qhash --url pool.example.com:8610 --user WALLET.WORKER --gpu 0

# Debug mode
rust-miner --algo qhash --url pool.example.com:8610 --user WALLET.WORKER --debug

# Help
rust-miner --help
```

**Example with Qubitcoin pool**:
```bash
./target/release/rust-miner \
  --algo qhash \
  --url qubitcoin.luckypool.io:8610 \
  --user bc1qacadts4usj2tjljwdemfu44a2tq47hch33fc6f.RIG-1 \
  --pass x
```

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
