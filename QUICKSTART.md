# 🚀 Quick Start - rust-miner (CUDA-only)

## TL;DR - Começar Agora

```bash
# 1) Instalar Rust (se necessário)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 2) Instalar CUDA Toolkit (12.x+)
# (consulte SETUP.md para instruções da sua distro)

# 3) Build
cargo build --release

# 4) Rodar com a sua pool
./target/release/rust-miner \
	--algo qhash \
	--url qubitcoin.luckypool.io:8610 \
	--user WALLET.WORKER \
	--pass x
```

## Comandos Essenciais

### Build
```bash
# CUDA (único backend)
cargo build --release
```

### Testes
```bash
# Testes básicos
cargo test

# Testes com output
cargo test -- --nocapture
```

### Performance
```bash
# Profiling com flamegraph
cargo flamegraph
 
# Verificar otimizações
cargo bloat --release
```

### Desenvolvimento
```bash
# Auto-rebuild
cargo watch -x 'build'

# Linter
cargo clippy

# Formatação
cargo fmt
```

## Estrutura do Projeto

```
rust-miner/
├── src/
│   ├── main.rs             # Orquestração de mineração
│   ├── backend.rs          # Trait MiningBackend (despacho dinâmico)
│   ├── cuda/
│   │   ├── mod.rs          # CUDA wrapper
│   │   ├── qhash.cu        # Kernel QHash (CUDA)
│   │   └── qhash_backend.rs# Backend QHash (trait impl)
│   ├── mining.rs           # Utilitários (merkle, nbits, hex)
│   ├── stratum/            # Cliente Stratum V1
│   └── gpu/                # Detecção de GPU
├── Cargo.toml
├── SETUP.md
└── QUICKSTART.md
```

## Algoritmo Suportado

- qhash

## Verificar Instalação

```bash
# Rust
rustc --version
cargo --version

# CUDA (prioritário)
nvcc --version
nvidia-smi

# Performance tools
perf --version
valgrind --version
```

## Observações

- Apenas CUDA é suportado (GPU NVIDIA obrigatória)

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
