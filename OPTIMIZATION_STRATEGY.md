# Estratégia de Otimização de Hashrate: CUDA Kernel Tuning

## 📊 Estado Atual
- **Hashrate Atual**: ~37 MH/s baseline (GTX 1660 SUPER, CC 7.5)
- **Picos Observados**: 12-253 MH/s (kernel variável)
- **Target**: ~200+ MH/s sustentável (5-6x de melhoria)
- **Gargalo Principal**: Kernel execution time ~480ms por bloco

## 🎯 Objetivos Realistas
1. **Reduzir latência do kernel** através de loop unrolling mais agressivo
2. **Aumentar occupancy** do warp scheduler (atualmente 512 threads/block)
3. **Otimizar memory access patterns** em quantum_simulation
4. **Melhorar instruction-level parallelism (ILP)** no SHA256

## 📚 Abordagens Testadas

### ❌ Tentativa 1: PTX Inline Assembly (FALHADA)
- Implementado `shf.r.wrap.b32` para rotações PTX
- **Resultado**: Hashrate caiu de 37 MH/s para 12 MH/s
- **Razão**: Overhead de multiple function calls vs. macro expansion
- **Lição**: Compiler C++ do NVRTC já otimiza bem; assembly inline não garantida melhoria

## ✅ Abordagens Comprovadas (A Implementar)

### ❌ Fase 1: Loop Unrolling & PTX Assembly (FALHADA - CANCELADA)
- **Testado**: Manual unrolling (8 rounds): 6 MH/s (83% regressão)
- **Testado**: `#pragma unroll 4`: 7.81 MH/s (78% regressão) 
- **Testado**: Baseline `#pragma unroll 64`: 37 MH/s (estável, MANTIDO)
- **Razão da falha**: 
  - NVRTC compiler já otimiza loop bem; mais unroll = MENOS performance
  - Menos unroll = mais register reuse, menos occupancy
  - PTX inline assembly: overhead de function calls > benefit (12 MH/s, -67% vs baseline)
- **Conclusão**: Gargalo **NÃO** é SHA256_transform latency
- **Lição Crítica**: Compiler já está MUITO bem otimizado; otimizações manual degradam

### ✅ Fase 2: Profiling para Identificar Gargalo Real
Usar `nvprof` para medir:
- **Occupancy**: Percentage de warps ativos (target: 75%+)
- **Memory Bandwidth**: GB/s vs. peak GPU (GTX 1660: ~336 GB/s)
- **Shared Memory Bank Conflicts**: Em quantum_simulation
- **Register Pressure**: Spills/reloads (atualmente ~100 registros/thread?)
- **L1/L2 Cache Hit Rates**: Memory access efficiency
```cuda
// Em quantum_simulation: substituir shared_expectations[] com shuffles
float my_val = expectations[tid];
float left = __shfl_up_sync(0xFFFFFFFF, my_val, 1);
float right = __shfl_down_sync(0xFFFFFFFF, my_val, 1);
// Reduz bank conflicts em shared memory
```

### ✅ Fase 3: Register Pressure Tuning
- Verificar `-Xptxas -O3` flag
- Usar `maxregcount` para balancear occupancy vs. performance
- Target: 64-96 registros/thread (máximo para SM 7.5)

### ✅ Fase 3: Warp-Level Optimization com __shfl_sync

## 🔬 Métricas de Sucesso

| Métrica | Baseline | Target | Método |
|---------|----------|--------|--------|
| Hashrate | 37 MH/s | 150+ MH/s | Loop unroll + warp opt |
| Kernel Time | 480 ms | 100-150 ms | Reduzir branch divergence |
| Occupancy | 50% | 75%+ | Register tuning |
| Power Eff | 0.9 MH/W | 2.5+ MH/W | Instruction/clock |

## 📋 Plano de Ação (ATUALIZADO - PROFILING FIRST)

### ❌ [PASSO 1] ~~Loop Unrolling Manual~~ (COMPLETADO - FALHOU)
- ✅ Testado: manual unroll, pragma unroll 4
- ✅ Resultado: Ambos degradaram performance 78-83%
- ✅ Decisão: MANTER baseline `#pragma unroll 64`
- ⏭️ **Próximo**: Profiling para identificar REAL bottleneck

### ✅ [PASSO 2] Profiling Detalhado (PRÓXIMO - CRÍTICO)
**Comandos a executar**:
```bash
# Medir occupancy e memory bandwidth
nvprof --metrics achieved_occupancy,sm_efficiency,memory_load_gld_efficiency \
  ./target/release/rust-miner --algo qhash --url qubitcoin.luckypool.io:8610 \
  --wallet bc1qacadts4usj2tjljwdemfu44a2tq47hch33fc6f --worker RIG-1 --pool-pass x

# Medir bank conflicts em shared memory
nvprof --events shared_load_bank_conflict,shared_store_bank_conflict \
  ./target/release/rust-miner ...

# Verificar register pressure
nvprof --metrics local_load,local_store,register_replay \
  ./target/release/rust-miner ...
```
**Métricas a coletar**:
- [ ] Occupancy % (target: 75%+)
- [ ] Memory bandwidth utilization (vs. peak 336 GB/s GTX 1660)
- [ ] Shared memory bank conflicts (em quantum_simulation)
- [ ] Register spills/reloads
- [ ] L1/L2 cache hit rates
- [ ] IPC (instructions per clock)

**Suspeitas atuais** (ordem de probabilidade):
1. **Shared memory access patterns** em quantum_simulation (256 floats, potencial bank conflicts)
2. **Low occupancy** devido register pressure (w[64] na sha256_transform)
3. **Memory bandwidth bottleneck** em data transfers entre threads
4. **Branch divergence** em quantum operations ou validation

### ✅ [PASSO 3] Otimizar Gargalo Identificado (DEPOIS DO PROFILING)
**Cenários possíveis**:
- **Se shared memory bank conflicts**: Implementar __shfl_sync para quantum_simulation
- **Se low occupancy**: Reduzir w[64] local array (usar global memory ou restructure)
- **Se memory bandwidth**: Batch nonces melhor, prefetch anticipation
- **Se branch divergence**: Substituir condicionais com arithmetic

### ✅ [PASSO 4] Validação & Benchmark
- [ ] Run 5x 40-segundo benchmarks após cada mudança
- [ ] Documentar: hashrate min/max/avg, kernel time, power
- [ ] Comparar vs. 37 MH/s baseline
- [ ] Se melhora >= 10%: commit e continua
- [ ] Se melhora < 10%: revert e próxima ideia

### ✅ [PASSO 5] Git Workflow
- [ ] Commit cada mudança com medições completas
- [ ] Branch: `optimization/profiling-phase`
- [ ] MR com benchmark comparisons
- [ ] Merge apenas se baseline 37 MH/s não regrediu

## ⚖️ Trade-offs

| Abordagem | Pros | Cons |
|-----------|------|------|
| PTX Inline | Total control, max perf potential | Complexo, difícil debug, breaking changes |
| __shfl_sync | Pronto, documentado, comprovado | Memória ainda usada em outros contextos |
| Loop Unroll Manual | Simples, predictável | Mais código, menos flexible |

## � Plano de Ação (ATUALIZADO)

### [PASSO 1] Implementar Loop Unrolling Manual (Esta Semana)
- [ ] Reescrever sha256_transform com 8 rounds por iteração
- [ ] Testar compilação sem warnings
- [ ] Benchmark: comparar vs. #pragma unroll 64

### [PASSO 2] Warp-Level Optimizations (Próxima)  
- [ ] Substituir shared_expectations[] com __shfl_sync
- [ ] Medir redução de shared memory bank conflicts
- [ ] Benchmark: comparar vs. shared memory baseline

### [PASSO 3] Register Pressure Tuning (Validação)
- [ ] Profiling com `nvprof --metrics achieved_occupancy`
- [ ] Ajustar maxregcount conforme necessário
- [ ] Final benchmark com ambos otimizações

### [PASSO 4] Git Commit & Documentation
- [ ] Commit cada mudança com medições
- [ ] Documentar resultados reais vs. targets
- [ ] Push para mainline

## ⚖️ Trade-offs

| Abordagem | Pros | Cons |
|-----------|------|------|
| PTX Inline | Total control, max perf potential | Complexo, difícil debug, breaking changes |
| __shfl_sync | Pronto, documentado, comprovado | Memória ainda usada em outros contextos |
| Loop Unroll Manual | Simples, predictável | Mais código, menos flexible |

## �️ Ferramentas de Profiling

```bash
# Profiling detalhado
nvprof --metrics all ./target/release/rust-miner ...

# Análise de occupancy
nvprof --events all ./target/release/rust-miner ...

# PTX inspection
nvdisasm -c sm_75 kernel.ptx | grep -E "sha256|quantum"
```

---

**Status**: Pronto para [PASSO 1] implementação de Loop Unrolling Manual
