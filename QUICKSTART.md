# 🚀 Quick Start - rust-miner

## TL;DR - Começar Agora

```bash
# 1. Instalar tudo automaticamente
cd /home/regis/develop/rust-miner
bash setup.sh

# 2. Após instalação do Rust, recarregar ambiente
source ~/.cargo/env

# 3. Build com CUDA (recomendado para GTX 1660 SUPER)
cargo build --release --features cuda

# 4. Rodar
./target/release/rust-miner
```

## Comandos Essenciais

### Build
```bash
# CUDA (padrão, melhor performance)
cargo build --release --features cuda

# Todos os backends (auto-detect)
cargo build --release --features all-backends

# CPU apenas (desenvolvimento)
cargo build --release --features cpu-only
```

### Testes
```bash
# Testes básicos
cargo test

# Testes com CUDA
cargo test --features cuda

# Testes com output
cargo test -- --nocapture
```

### Performance
```bash
# Benchmarks CUDA
cargo bench --features cuda

# Profiling com flamegraph
cargo flamegraph

# Verificar otimizações
cargo bloat --release --features cuda
```

### Desenvolvimento
```bash
# Auto-rebuild on changes
cargo watch -x 'build --features cuda'

# Linter
cargo clippy --features cuda

# Formatação
cargo fmt
```

## Estrutura do Projeto

```
rust-miner/
├── .github/
│   └── copilot-instructions.md    # ⭐ Guia completo para AI/Dev
├── src/
│   ├── main.rs                    # Entry point
│   ├── mining/                    # Mining engine
│   │   ├── engine.rs
│   │   ├── cuda.rs               # ⭐ CUDA backend
│   │   ├── opencl.rs             # Fallback
│   │   └── cpu.rs                # CPU fallback
│   └── blockchain/                # Blockchain interface
├── Cargo.toml                     # Dependencies + features
├── SETUP.md                       # Setup detalhado
├── setup.sh                       # Setup automatizado
└── QUICKSTART.md                  # Este arquivo
```

## Features do Cargo.toml

```toml
[features]
default = ["cuda"]                 # ⭐ CUDA por padrão
cpu-only = []                      # CPU apenas
cuda = ["dep:cudarc", ...]         # NVIDIA (PRIMARY)
opencl = ["dep:ocl"]               # AMD/Intel (FALLBACK)
all-backends = ["cuda", "opencl"]  # Todos
```

## Verificar Instalação

```bash
# Rust
rustc --version
cargo --version

# CUDA (prioritário)
nvcc --version
nvidia-smi

# OpenCL (fallback)
clinfo | head -20

# Performance tools
perf --version
valgrind --version
```

## Prioridade de Backends

```
1️⃣  CUDA     (GTX 1660 SUPER → ~26 MH/s Ethash)
2️⃣  OpenCL   (Fallback → ~22 MH/s)
3️⃣  CPU      (12 threads → ~0.5 MH/s)
```

## Links Importantes

- **Copilot Instructions**: `.github/copilot-instructions.md` - Patterns e best practices
- **Setup Detalhado**: `SETUP.md` - Guia completo de instalação
- **cudarc Docs**: https://docs.rs/cudarc - CUDA para Rust
- **Rust Book**: https://doc.rust-lang.org/book/ - Aprender Rust

## Troubleshooting Rápido

### CUDA não encontrado
```bash
# Instalar CUDA Toolkit
sudo apt install cuda-toolkit-13-0

# Adicionar ao PATH
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
```

### Build lento
```bash
# Usar todos os cores
export CARGO_BUILD_JOBS=12

# Ou adicionar em ~/.cargo/config.toml
[build]
jobs = 12
```

### GPU não detectada
```bash
# Verificar driver NVIDIA
nvidia-smi

# Testar CUDA
cuda-samples/deviceQuery

# Verificar OpenCL
clinfo
```

## Próximos Passos

1. ✅ Executar `setup.sh`
2. ✅ Verificar instalação (comandos acima)
3. 📖 Ler `.github/copilot-instructions.md` para patterns
4. 💻 Implementar mining engine em `src/mining/`
5. 🎮 Adicionar CUDA kernel em `src/mining/cuda.rs`
6. 🧪 Criar testes e benchmarks
7. ⚡ Otimizar performance

---
**Hardware**: AMD Ryzen 5 5600X + GTX 1660 SUPER = Excelente para mining! 🚀
