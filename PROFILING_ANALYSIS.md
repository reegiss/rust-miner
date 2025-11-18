# Análise de Profiling - QHash CUDA Kernel

## 📊 Observações do Código

### 1. Shared Memory Bank Conflicts (CRÍTICO)

**Localização**: `quantum_simulation()` linhas 240-290

```cuda
__shared__ float shared_expectations[256]; // 256 threads max

if (tid < N_QUBITS) {
    shared_expectations[tid] = 0.0f;  // Thread 0→bank 0, 1→bank 1, ...
}
__syncthreads();

// CNOT operation
if (tid < N_QUBITS - 1) {
    float exp_i = shared_expectations[tid];      // Load from bank tid
    float exp_ip1 = shared_expectations[tid + 1]; // Load from bank tid+1
    // PROBLEMA: Sequential access = 2 32-bit loads per 4 bancos
    // GTX 1660 (CC 7.5) = 32 bancos, 4 bytes cada
    // Thread 0: loads bank 0, 1
    // Thread 1: loads bank 1, 2
    // etc. = NO BANK CONFLICT (lucky!)
}
```

**Análise**: 
- Float = 4 bytes, shared memory = 32 banks (CC 7.5)
- Stride-1 access em float = sem bank conflict
- **Mas**: Padding pode ocorrer se array > 2KB
- 256 floats = 1024 bytes = 32.7KB (ACIMA DO LIMITE!)
- **Problema Real**: Apenas 16 threads usam shared_expectations[], 240 threads DESPERDIÇADAS

### 2. Warp Divergence (MODERADO)

**Localização**: `qhash_mine()` linhas 320-350

```cuda
if (tid < N_QUBITS) {          // tid 0-15: active, 16-511: inactive
    shared_expectations[tid] = 0.0f;
}
__syncthreads();

if (tid < N_QUBITS - 1) {      // tid 0-14: active, 15-511: inactive
    // CNOT operations
}
__syncthreads();
```

**Impacto**:
- 512 threads/block, apenas 16 ativas em quantum_simulation
- 496 threads = **96.9% DESPERDÍCIO**
- Kernel time = 480ms = MUITO LONGO para operações simples

### 3. Register Pressure

**Localização**: `qhash_mine()` registradores locais

```cuda
uint8_t header[80];      // 80 bytes = 20 uint32_t registers
uint8_t hash1[32];       // 32 bytes = 8 uint32_t registers
uint8_t nibbles[64];     // 64 bytes = 16 uint32_t registers
float expectations[16];  // 16 floats = 16 uint32_t registers
uint8_t quantum_output[32]; // 32 bytes = 8 registers
uint8_t final_input[64]; // 64 bytes = 16 registers
// TOTAL: ~84 registers/thread = ALTO
```

**GTX 1660 (CC 7.5) limit**: 255 registers/SM
- SM capacity: 2560 threads (8 warps × 32 threads = 256 threads/warp × 10 warps)
- Com 84 regs/thread: 2560/84 = 30 threads máximo
- **Occupancy**: ~30/32 = 93% PER WARP = mas só 30 threads ativos!
- Com 512 threads/block: 512/2560 = 20% da capacidade do SM!

### 4. Kernel Execution Bottleneck

**Tempo observado**: 480ms por kernel é MUITO LONGO

Análise:
- SHA256 (2x): ~40 operações × 2 = 80 ops
- Quantum sim: ~50 ops (rotações + CNOT + tanh)
- Comparações: ~40 ops
- **Total**: ~170 operações
- GTX 1660: 1530-6801 MHz, FP32: ~15.2 TFLOPS (single precision)
- Tempo esperado (CPU-bound): 170 ops / 15.2T ops/sec = **11 ns**
- **Observado**: 480ms = **43.6 BILHÕES de vezes mais lento!**

**Conclusão**: Kernel NÃO é CPU-bound. É **MEMORY-BOUND** ou **STALL-BOUND**.

### 5. Memory Access Patterns

**SHA256 transform**:
- `w[64]` array em stack (registrador/local memory)
- 64 × 4 bytes = 256 bytes/thread
- Local memory = muito lento (DRAM latency)

**Quantum simulation**:
- `expectations[16]` em registrador (rápido)
- `shared_expectations[256]` em shared mem (medium)
- Acesso sequencial OK, mas tamanho > limite

**Final hash**:
- `final_input[64]` em registrador
- `final_hash[32]` em registrador

### 6. Synchronization Overhead

```cuda
for (int l = 0; l < N_LAYERS; l++) {
    // Layer 1
    __syncthreads();  // Tread 16-511 NÃO têm dados para sincronizar
    // Layer 2
    __syncthreads();  // NOVAMENTE: desperdício
}
```

**Problema**: `__syncthreads()` bloqueia TODAS threads mesmo que 496 façam nada.

## 🎯 Diagnóstico Final

| Métrica | Valor | Avaliação |
|---------|-------|-----------|
| Occupancy (teórico) | 20% (512/2560 threads) | ❌ BAIXO |
| Register usage | 84 regs/thread | ⚠️ ALTO |
| Shared memory | 1KB (array) + 108B (header/target) = 1.1KB | ✅ OK |
| Warp divergence | 96.9% threads inativos em quantum_sim | ❌ CRÍTICO |
| Memory access | Sequential (OK), mas local mem spill | ⚠️ MEDIUM |
| Sync overhead | 2+ syncthreads per kernel | ⚠️ HIGH |

**GARGALO PRINCIPAL**: Warp divergence + ocupação baixa
- Apenas 16 de 512 threads fazem trabalho útil
- Resto = travado em syncthreads()
- Kernel time = esperando threads inativas

## ✅ Estratégia de Otimização

### Fase 1: Reduzir Warp Divergence (FÁCIL, +50% esperado)
- Reducir threads/block de 512 → 64 (múltiplo de 32)
- Apenas 16 threads precisam rodar, 48 de buffer
- Remove desperdício de sincronização

### Fase 2: Reduzir Register Pressure (MÉDIO, +20% esperado)
- Mover `w[64]` para shared memory (mas cuidado com bank conflicts)
- Ou usar compressão: SHA256 state em 4×64-bit em vez 8×32-bit

### Fase 3: Remover Local Memory Spills (MÉDIO, +15% esperado)
- Verificar se `header[80]`, `nibbles[64]` causam spills
- Usar `__restrict__` para memory aliasing

### Fase 4: Multi-threading da Quantum Simulation (AVANÇADO, +200% esperado)
- Distribuir 16 qubits em 16 threads em vez de 1
- Fazer quantum_sim com warp shuffle reduction

## 📈 Target: 37 MH/s → 150+ MH/s (4x melhoria)

Estimativa:
- Fase 1 (warp divergence): 37 × 1.5 = **55 MH/s**
- Fase 2 (register): 55 × 1.2 = **66 MH/s**
- Fase 3 (local mem): 66 × 1.15 = **76 MH/s**
- Fase 4 (multi-threading): 76 × 2.0 = **152 MH/s** ✅

---

**Status**: Pronto para implementação Fase 1 (mais seguro, menor risco de regressão)
