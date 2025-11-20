# rust-miner

High-performance GPU cryptocurrency miner written in Rust. CUDA-only, future-proof architecture with dynamic algorithm dispatch.

• Cross-Platform: Linux and Windows
• GPU Required: NVIDIA GPU with CUDA (no CPU mining)

## ⚡ Features

- CUDA-only backend (NVIDIA GPUs)
- Dynamic algorithm dispatch via trait-based backend
- QHash (quantum PoW) implemented on CUDA
- Stratum V1 protocol (subscribe, notify, submit)
- Kernel returns final hash directly (no CPU recomputation)
- Adaptive batch sizing (targets ~700–900 ms per kernel)
- Efficient non-blocking driver polling (~6% CPU)

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

⚠️ This application requires an NVIDIA GPU with CUDA support. Systems without compatible NVIDIA hardware cannot mine.

## 🔧 Build & Test

```bash
# Build (CUDA required)
cargo build --release

# Dev build
cargo build

# Run tests
cargo test
```

Note: CUDA is the only supported backend. CPU mining is not available.

## 📊 Performance (Current)

- GTX 1660 SUPER (CUDA) — QHash: ~295 MH/s
- Stable hashrate with analytical QHash implementation
- Shares successfully submitted to Qubitcoin pool

Notes:
- Performance depends on CUDA version/driver and power limits
- Analytical QHash algorithm uses lookup table for quantum simulation
- No CPU mining support (CUDA GPU required)

## 🏗️ Architecture

```
rust-miner/
├── src/
│   ├── main.rs              # Entry point, mining orchestration
│   ├── cli.rs               # Command-line interface
│   ├── backend.rs           # MiningBackend trait (dynamic algorithm dispatch)
│   ├── cuda/
│   │   ├── mod.rs           # CUDA wrapper and device mgmt
│   │   ├── qhash.cu         # QHash kernel (CUDA C++)
│   │   └── qhash_backend.rs # QHash backend implementing MiningBackend
│   ├── mining.rs            # Helpers: merkle, targets, conversions
│   ├── stratum/
│   │   ├── client.rs        # Stratum V1 client
│   │   └── protocol.rs      # Protocol types
│   └── gpu/
│       ├── mod.rs           # GPU detection
│       └── cuda.rs          # CUDA device info
└── .github/
  └── copilot-instructions.md  # Development guidelines
```

Key ideas:
- CUDA-only backend (NVIDIA GPUs only)
- Trait-based backend abstraction (future algorithms plug-in)
- Kernel returns (nonce, hash) directly
- Analytical QHash implementation with lookup table
- Efficient non-blocking driver polling

Supported algorithms:
- qhash

## 🧪 Testing

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# GPU-specific tests require hardware

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
rust-miner --algo qhash --url qubitcoin.luckypool.io:8610 --user bc1qacadts4usj2tjljwdemfu44a2tq47hch33fc6f.RIG-1 --pass x

# With specific GPU
rust-miner --algo qhash --url qubitcoin.luckypool.io:8610 --user bc1qacadts4usj2tjljwdemfu44a2tq47hch33fc6f.RIG-1 --gpu 0

# Debug mode
rust-miner --algo qhash --url qubitcoin.luckypool.io:8610 --user bc1qacadts4usj2tjljwdemfu44a2tq47hch33fc6f.RIG-1 --debug

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

Tip: backend is initialized before pool connect to fail-fast on unsupported --algo.

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
- [Qubitcoin LuckyPool](https://luckypool.io/)

---

**Built with ❤️ using Rust**
