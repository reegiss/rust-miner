#!/bin/bash
# Setup script for rust-miner development environment
# Run with: bash setup.sh

set -e  # Exit on error

echo "============================================"
echo "🦀 rust-miner Development Setup"
echo "============================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check functions
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 is installed"
        return 0
    else
        echo -e "${RED}✗${NC} $1 is not installed"
        return 1
    fi
}

# Step 1: Check prerequisites
echo "📋 Checking prerequisites..."
echo ""

check_command gcc || echo "  Install with: sudo apt install build-essential"
check_command git || echo "  Install with: sudo apt install git"

echo ""

# Step 2: Install Rust
echo "🦀 Installing Rust toolchain..."
echo ""

if ! check_command rustc; then
    echo "Installing Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    echo -e "${GREEN}✓${NC} Rust installed successfully"
else
    echo "Rust already installed, updating..."
    rustup update
fi

echo ""

# Step 3: Install Rust components
echo "📦 Installing Rust components..."
rustup component add clippy rustfmt rust-analyzer 2>/dev/null || true
echo -e "${GREEN}✓${NC} Components installed"
echo ""

# Step 4: Install GPU support (CUDA priority)
echo "🎮 Setting up GPU support..."
echo ""
echo "⭐ Priority: CUDA (NVIDIA) for maximum performance"
echo "   Fallback: OpenCL for compatibility"
echo ""

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    echo "✓ NVIDIA GPU detected: $GPU_NAME"
    echo ""
    
    # Install CUDA Toolkit
    echo "Installing CUDA Toolkit (primary backend)..."
    if ! command -v nvcc &> /dev/null; then
        echo "CUDA compiler not found. Installing CUDA Toolkit 13.0..."
        echo "This may take a while..."
        
        # Note: User may need to install manually if this fails
        sudo apt update
        sudo apt install -y cuda-toolkit-13-0 2>/dev/null || \
        echo -e "${YELLOW}⚠${NC} CUDA Toolkit installation requires manual setup. See SETUP.md"
        
        # Add to PATH
        if ! grep -q "cuda-13.0" ~/.bashrc; then
            echo 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> ~/.bashrc
            echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
            export PATH=/usr/local/cuda-13.0/bin:$PATH
            export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
        fi
    else
        echo -e "${GREEN}✓${NC} CUDA Toolkit already installed: $(nvcc --version | grep release | cut -d' ' -f5)"
    fi
    
    echo ""
    echo "Installing OpenCL as fallback..."
fi

# Install OpenCL support (fallback or non-NVIDIA)
if ! check_command clinfo; then
    echo "Installing OpenCL runtime..."
    sudo apt update
    sudo apt install -y ocl-icd-libopencl1 opencl-headers clinfo
    
    # Install NVIDIA OpenCL support if NVIDIA GPU detected
    if command -v nvidia-smi &> /dev/null; then
        echo "Installing NVIDIA OpenCL support (fallback backend)..."
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d'.' -f1)
        sudo apt install -y nvidia-opencl-icd-${DRIVER_VERSION} 2>/dev/null || \
        sudo apt install -y nvidia-opencl-icd || \
        echo -e "${YELLOW}⚠${NC} Could not install NVIDIA OpenCL, may need manual installation"
    fi
    
    echo -e "${GREEN}✓${NC} OpenCL support installed (fallback)"
else
    echo -e "${GREEN}✓${NC} OpenCL already available"
fi

echo ""

# Step 5: Install development tools
echo "🔧 Installing development tools..."
echo ""

# Performance profiling tools
sudo apt install -y linux-tools-common linux-tools-generic 2>/dev/null || true
sudo apt install -y valgrind 2>/dev/null || true

# Cargo tools
echo "Installing cargo extensions..."
cargo install cargo-watch --quiet 2>/dev/null || echo "cargo-watch already installed"
cargo install flamegraph --quiet 2>/dev/null || echo "flamegraph already installed"

echo -e "${GREEN}✓${NC} Development tools installed"
echo ""

# Step 6: Configure Cargo
echo "⚙️  Configuring Cargo for optimal performance..."
mkdir -p ~/.cargo

cat > ~/.cargo/config.toml << 'EOF'
[build]
jobs = 12

[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "target-cpu=native"]

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
strip = true
EOF

echo -e "${GREEN}✓${NC} Cargo configured"
echo ""

# Step 7: Initialize project if not already done
echo "📁 Initializing project structure..."
echo ""

if [ ! -f "Cargo.toml" ]; then
    cargo init --name rust-miner
    echo -e "${GREEN}✓${NC} Project initialized"
else
    echo -e "${GREEN}✓${NC} Project already initialized"
fi

# Create .gitignore if not exists
if [ ! -f ".gitignore" ]; then
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
*.log
EOF
    echo -e "${GREEN}✓${NC} .gitignore created"
fi

echo ""

# Step 8: Test build
echo "🔨 Testing build..."
echo ""

if cargo build 2>&1 | grep -q "Finished"; then
    echo -e "${GREEN}✓${NC} Build successful"
else
    echo -e "${YELLOW}⚠${NC} Build test skipped or failed (normal for empty project)"
fi

echo ""

# Summary
echo "============================================"
echo "✅ Setup Complete!"
echo "============================================"
echo ""
echo "Environment Summary:"
echo "  • Rust: $(rustc --version | cut -d' ' -f2)"
echo "  • Cargo: $(cargo --version | cut -d' ' -f2)"
echo "  • CPU: $(nproc) cores available"

if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    echo "  • GPU: $GPU_NAME (NVIDIA)"
    
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep release | cut -d' ' -f5 | cut -d',' -f1)
        echo "  • CUDA: $CUDA_VERSION (PRIMARY backend ⭐)"
    else
        echo "  • CUDA: Not installed (install manually for best performance)"
    fi
fi

if command -v clinfo &> /dev/null; then
    OPENCL_DEVICES=$(clinfo 2>/dev/null | grep -c "Device Name" || echo "0")
    echo "  • OpenCL devices: $OPENCL_DEVICES (fallback backend)"
fi

echo ""
echo "Backend Priority: CUDA → OpenCL → CPU"
echo ""
echo "Next steps:"
echo "  1. Review .github/copilot-instructions.md for coding guidelines"
echo "  2. Review SETUP.md for detailed environment info"
echo "  3. Start coding: edit src/main.rs"
echo "  4. Build with CUDA: cargo build --release --features cuda"
echo "  5. Build with all backends: cargo build --release --features all-backends"
echo "  6. Run tests: cargo test --features cuda"
echo ""
echo "Happy mining! 🚀"
