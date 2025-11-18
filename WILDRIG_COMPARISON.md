# WildRig QHash Benchmark vs rust-miner

## Conhecidos (Fatos)
- **WildRig**: 500 MH/s em GTX 1660 SUPER com QHash
- **rust-miner**: 37 MH/s em GTX 1660 SUPER com QHash
- **Diferença**: 13.5x (muito grande para explicar por simples otimizações)

## GPU Specs GTX 1660 SUPER (CC 7.5)
- **CUDA cores**: 1408
- **Memory**: 6 GB GDDR6
- **Memory bandwidth**: 336 GB/s (192-bit bus, 12 Gb/s)
- **Max threads per SM**: 1024
- **Max blocks per SM**: 8

## Análise Teórica

### Limite Absoluto de Throughput (impossível na prática)
```
Peak throughput = cores × clock × ops_per_clock
               = 1408 × 1536 MHz × 2 (fused ops) 
               = ~4.3 TFLOPS
```

### QHash Hashrate Teórico
```
Uma hash QHash requer:
- SHA256 inicial: ~300 cycles (com latência)
- Quantum sim: ~200 cycles (local math)
- SHA256 final: ~300 cycles
- Total: ~800 cycles por hash

Se 100% de cores fossem ativos:
Hashrate = (1408 cores × 1536 MHz) / 800 cycles
         = ~2.7 GH/s teórico máximo
         = ~2700 MH/s
```

**Logo, 500 MH/s é ~18.5% do teórico máximo. Realista.**

### Análise do Gargalo em rust-miner

**Atual**: 37 MH/s = ~1.4% do teórico máximo

**Possíveis causas da sub-utilização**:

1. **Occupancy** (87.5% ociosa com 128 threads/block)
   - Apenas 12.5% dos cores ativos
   - GPU sentada esperando

2. **Memory Transfers** (improvável)
   - Apenas 76 + 32 bytes entrada, 32 bytes saída
   - Fully coalesced em meu código
   - Não há memory bottleneck

3. **Algorithm Correctness** (improvável)
   - SHA256 usa macros (compilador otimiza bem)
   - Quantum sim é simples (16 qubits, 2 layers)
   - Code path deve ser correto (funciona)

4. **Kernel Launch Overhead** (possível)
   - Estou relançando kernel para cada batch
   - Cada launch = overhead CPU
   - Se batch_size pequeno, overhead domina

5. **Compilation Flags** (muito provável)
   - NVRTC pode estar compilando com `-O0`
   - Sem loop unrolling automático
   - Sem instruction scheduling

## Diagnosticar o Real Gargalo

### Teste 1: Aumentar Occupancy
```
Mudar threads_per_block: 128 → 256 ou 512
Esperado: Se 37 MH/s → 70-100 MH/s = occupancy era gargalo
          Se 37 MH/s → ~37 MH/s = occupancy não era problema
```

### Teste 2: Profiling com nsys
```bash
nsys profile --stats=true ./target/release/rust-miner | grep qhash_mine
# Mostrará: occupancy, memory bandwidth utilization, kernel time
```

### Teste 3: NVRTC Compilation Flags
```
Adicionar flags de otimização:
- "-O3" (máxima otimização)
- "-Xptxas -O3" (PTX assembler optimization)
- "-use_fast_math" (relaxar precisão para speed)
```

### Teste 4: Batch Size
```
Meu código: Quantos nonces por batch?
WildRig pode estar usando batch_size = 10M nonces
Meu código pode estar usando batch_size = 100k (overhead alto)
```

## Diferenças Arquiteturais Entre rust-miner e WildRig

### O que WildRig provavelmente faz DIFERENTE:

1. **Kernel Batching**
   - Uma única kernel launch por minuto
   - Processa MILHÕES de nonces por launch
   - Reduz overhead de launching

2. **Compilation Optimization**
   - `-O3` enabled
   - Loop unrolling agressivo
   - Instruction scheduling otimizado

3. **Memory Layout**
   - Dados em global memory, not registers
   - Coalesced access patterns
   - Shared memory para comunicação

4. **Thread Configuration**
   - Ocupancy pode estar em 50% (512 threads/block)
   - Mais blocos ativos simultaneamente

5. **Kernel Streamlining**
   - Remove redundant checks
   - Elimina branches
   - Inlines everything

6. **CUDA Stream Management**
   - Múltiplos streams paralelos
   - Overlapping compute + memory

## Recomendação: Teste Empiricamente

Antes de ler código WildRig ou fazer PTX assembly:

### FASE 1: Occupancy Test (5 minutos)
```bash
# Em src/cuda/mod.rs, trocar linha 63:
let threads_per_block = 256;  # ou 512
cargo build --release
# Medir hashrate com time ./target/release/rust-miner
# Se aumenta: occupancy era problema
# Se não muda: SHA256 é bottleneck
```

### FASE 2: Compilation Flags (10 minutos)
```bash
# Em build.rs ou cargo config, adicionar NVRTC flags:
"-O3", "-Xptxas", "-O3", "-use_fast_math"
cargo build --release
# Medir hashrate novamente
```

### FASE 3: Profile (15 minutos)
```bash
nsys profile -d 10 --stats=true ./target/release/rust-miner
# Ver: occupancy, SM utilization, memory bandwidth
```

Se ainda em 37 MH/s após esses, ENTÃO:
### FASE 4: PTX Assembly (proposto, mas provavelmente não necessário)

## Estimativa: Alcançar 500 MH/s

**Cenário Otimista**:
- Occupancy 12.5% → 50%: **4x ganho** (37 → 148 MH/s)
- Compilation flags: **1.2x ganho** (148 → 177 MH/s)
- Memory optimization: **1.5x ganho** (177 → 265 MH/s)
- Batch size + stream management: **2x ganho** (265 → 530 MH/s)

**Total possível**: ~530 MH/s ≈ 500 MH/s ✓

**Conclusão**: 500 MH/s é alcançável SEM PTX assembly inline!

## Ação Imediata

1. **NÃO** gaste tempo em PTX assembly ainda
2. **Primeiro**: Teste occupancy (256 threads/block)
3. **Segundo**: Compile com flags de otimização
4. **Terceiro**: Profile com nsys para confirmar bottleneck
5. **Quarto**: Se ainda baixo, ENTÃO investigar arquitetura (batching, streams, etc)

---

**Palpite**: rust-miner está usando 128 threads/block (12.5% ocupancy) e occupancy é o REAL gargalo, não SHA256.
