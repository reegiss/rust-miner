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

## 📋 Plano de Ação (ATUALIZADO - FASE 1 COMPLETA)

### ✅ [PASSO 1] Warp Divergence Fix (COMPLETO)
- ✅ Identificado: 96.9% de threads inativos em quantum_simulation
- ✅ Solução: Reduzir threads_per_block de 512 → 128
- ✅ Resultado: **kernel_time 480ms → 400ms (-16.7%)**
- ✅ Commits: 3bba943 (tuning), 65bf8bb (phase1 complete)

### ✅ [PASSO 2] Profiling Detalhado (COMPLETO)
- ✅ Método: Code analysis + nvidia-smi monitoring (nvprof não suporta CC 7.5)
- ✅ Descobertas: Gargalo NÃO é SHA256_transform (já bem otimizado)
- ✅ Gargalo REAL: Warp divergence em threads inativas
- ✅ Documentação: PROFILING_ANALYSIS.md, PHASE_1_RESULTS.md

### 🔄 [PASSO 3] Investigar Hashrate Variability (PRÓXIMO)
- [ ] Observação: kernel_time estável (400ms) mas hashrate varia (6-12 MH/s)
- [ ] Suspeita: Não é kernel, mas nonce distribution ou pool metrics
- [ ] Ação: Aumentar batch size de nonces ou investigar host thread scheduling
- [ ] Target: Entender por que 480ms kernel → 37 MH/s inicial vs 12 MH/s agora

### [PASSO 4] Memory Bandwidth Optimization (FUTURO)
- [ ] Investigar: w[64] array em stack pode causar local memory spills
- [ ] Solução: Move para global memory com cuidado de bandwidth
- [ ] Target: +10-15% se memory-bound

### [PASSO 5] IPC Improvement (FUTURO)
- [ ] Investigar: Instruction-level parallelism na SHA256 transform
- [ ] Solução: Rearranjar computações para maior ILP
- [ ] Target: +5-10% se compute-bound

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
