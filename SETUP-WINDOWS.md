# Windows Setup Guide - rust-miner (CUDA-only)

This guide covers Windows-specific setup for the rust-miner project.

## Prerequisites

### Required Software
- Windows 10/11 (64-bit)
- NVIDIA GPU with CUDA support (Pascal or newer recommended)
- Administrator access for driver installation

## Step 1: Install Rust

### Using rustup-init.exe (Recommended)
```powershell
# Download and run rustup-init.exe
# Visit: https://rustup.rs/

# Or use winget (Windows Package Manager)
winget install Rustlang.Rustup

# Verify installation
rustc --version
cargo --version
```

### Configure Rust
```powershell
# Install components
rustup component add clippy rustfmt rust-analyzer

# Set stable as default
rustup default stable
```

## Step 2: Install Visual Studio Build Tools

CUDA and many Rust crates require MSVC compiler:

```powershell
# Download Visual Studio Build Tools
# https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

# Required components:
# - C++ build tools
# - Windows 10/11 SDK
# - MSVC v143 or later
```

## Step 3: Install CUDA Toolkit (NVIDIA GPUs)

### Download CUDA Toolkit 13.0
1. Visit: https://developer.nvidia.com/cuda-downloads
2. Select: Windows > x86_64 > 10/11 > exe (local)
3. Download and run installer (~3GB)

### Verify CUDA Installation
```powershell
# Check CUDA compiler
nvcc --version

# Check NVIDIA driver
nvidia-smi

# Verify environment variables (should be set automatically)
echo $env:CUDA_PATH
echo $env:PATH | Select-String -Pattern "CUDA"
```

### Manual Environment Setup (if needed)
```powershell
# Add CUDA to PATH (adjust version as needed)
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"

# Make permanent (in PowerShell as Admin)
[Environment]::SetEnvironmentVariable("CUDA_PATH", $env:CUDA_PATH, "Machine")
[Environment]::SetEnvironmentVariable("PATH", "$env:CUDA_PATH\bin;$env:PATH", "Machine")
```

## Step 4: Install Development Tools

### Git for Windows
```powershell
# Using winget
winget install Git.Git

# Or download from: https://git-scm.com/download/win
```

### Optional Tools
```powershell
# Windows Terminal (better than cmd/powershell)
winget install Microsoft.WindowsTerminal

# PowerShell 7+
winget install Microsoft.PowerShell

# Chocolatey (package manager)
# Run in PowerShell as Admin:
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

## Step 6: Build the Project

```powershell
# Clone or navigate to project
cd C:\Users\YourName\Projects\rust-miner

# Build (CUDA-only)
cargo build --release --features cuda

# Run
.\target\release\rust-miner.exe
```

## Troubleshooting

### CUDA Not Found
```powershell
# Verify CUDA_PATH
echo $env:CUDA_PATH

# Check if nvcc is in PATH
where.exe nvcc

# Reinstall CUDA Toolkit if needed
```

### Link Errors (LNK1181, LNK2019)
```powershell
# Ensure Visual Studio Build Tools are installed
# Verify MSVC is in PATH
where.exe cl.exe

# May need to run from "Developer PowerShell for VS"
```

### "No CUDA device found"
```powershell
# Ensure your system has an NVIDIA GPU and drivers installed
nvidia-smi

# Verify CUDA Toolkit installation
nvcc --version
```

### Slow Compilation
```powershell
# Use link-time optimization (already in Cargo.toml)
# Use all CPU cores
$env:CARGO_BUILD_JOBS = [Environment]::ProcessorCount

# Or create ~/.cargo/config.toml:
# [build]
# jobs = 12  # adjust to your CPU
```

## Performance Profiling on Windows

### Windows Performance Analyzer
```powershell
# Install Windows SDK (includes WPA)
# Download from: https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/

# Record profile
wpr -start CPU -filemode

# Run your program
.\target\release\rust-miner.exe

# Stop recording
wpr -stop profile.etl

# Analyze with WPA
wpa profile.etl
```

### Alternative: cargo-flamegraph (cross-platform)
```powershell
cargo install flamegraph

# May need to run as Administrator
cargo flamegraph --features cuda
```

## WSL2 Alternative

For a Linux-like experience on Windows:

```powershell
# Enable WSL2
wsl --install

# Install Ubuntu
wsl --install -d Ubuntu-22.04

# Inside WSL2, follow Linux setup guide
wsl
bash setup.sh
```

**Note**: WSL2 supports CUDA with the NVIDIA WSL driver, but native Windows is recommended for best results.

## VSCode Setup (Windows)

### Recommended Extensions
```powershell
# Install via VSCode or command line
code --install-extension rust-lang.rust-analyzer
code --install-extension tamasfe.even-better-toml
code --install-extension vadimcn.vscode-lldb
code --install-extension serayuzgur.crates
```

### Configure rust-analyzer
Create `.vscode/settings.json`:
```json
{
  "rust-analyzer.cargo.features": ["cuda"],
  "rust-analyzer.checkOnSave.command": "clippy"
}
```

## Next Steps

1. ✅ Build project: `cargo build --release --features cuda`
2. ✅ Run tests: `cargo test`
3. ✅ Check performance: `cargo bench --features cuda`
4. 📖 Read [QUICKSTART.md](QUICKSTART.md) for usage
5. 📖 Read [.github/copilot-instructions.md](.github/copilot-instructions.md) for development

---

**Windows-specific notes**:
- Use PowerShell or Windows Terminal (not cmd.exe)
- Path separators: Use `std::path::PathBuf` in code (handles `\` automatically)
- Case-insensitive filesystem: Be consistent with file naming
- Line endings: Git handles CRLF ↔ LF conversion automatically
