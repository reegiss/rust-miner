# Diagnóstico de Hashrate Real - Teste Executar

## O Enigma
- **Kernel time**: ~400ms (medido em Phase 1)
- **Batch size**: 50M nonces
- **Hashrate teórico**: 50M / 0.4s = **125 MH/s**
- **Hashrate observado**: 37 MH/s
- **Eficiência**: 37 / 125 = **29.6%** (MUITO BAIXO)

## Possíveis Causas

1. **Kernel time NÃO é 400ms**
   - Realmente é ~1.35s (1350ms)?
   - Isso explicaria: 50M / 1.35s = 37 MH/s ✓

2. **Batch size NÃO é 50M**
   - Está adaptando para tamanho menor?
   - Se 13M nonces: 13M / 0.4s = 32.5 MH/s (próximo!)

3. **Pool Overhead**
   - Job switching = 50% do tempo?
   - Stratum latency = 200ms por job?

## Instruções para Teste

### TESTE 1: Ver o Real Kernel Time
```bash
# Rode com RUST_LOG=info
RUST_LOG=info ./target/release/rust-miner --url stratum+tcp://... --user ... 2>&1 | grep "GPU poll done"

# Você vai ver linhas como:
# GPU poll done: iters=16 elapsed_ms=400 batch_nonces=50000000 estimated_MH/s=125.0
# GPU poll done: iters=16 elapsed_ms=400 batch_nonces=50000000 estimated_MH/s=125.0
# ...

# Anote:
# - Real elapsed_ms para cada batch
# - Real batch_nonces processados
# - Estimated_MH/s calculado pelo kernel
```

### TESTE 2: Calcular Média de Hashrate de Kernel
```bash
# Rode por 1 minuto
RUST_LOG=info ./target/release/rust-miner --url ... --user ... 2>&1 | \
  grep "GPU poll done" | \
  awk '{print $NF}' | \
  awk '{sum+=$1; count++} END {print "Average kernel MH/s:", sum/count}'

# Isto mostrará a REAL taxa do kernel (ignorando overhead de pool)
# Se for 125 MH/s: kernel está ok, problema está em cima
# Se for 37 MH/s: kernel realmente é lento, precisa otimizar GPU
```

### TESTE 3: Medir Com threads_per_block = 256
```bash
# Já mudei para 256 em mod.rs
# Compile e rode:
cargo build --release
RUST_LOG=info ./target/release/rust-miner --url ... --user ... 2>&1 | \
  grep "GPU poll done" | \
  head -20
  
# Se elapsed_ms aumentar significativamente (>600ms):
#   → 256 threads/block não cabe em GPU
#   → Register spill ou occupancy problema
#   → Voltar para 128 ou tentar 192

# Se elapsed_ms diminuir (<300ms):
#   → Excelente! Ganho de performance
#   → Final hashrate deve ser 37 × (400/novo_ms)
```

## O que Esperar

### Cenário A: Kernel Time é realmente 400ms
```
Kernel: 50M nonces @ 400ms = 125 MH/s
Pool overhead: 37 MH/s observado
Conclusão: Você está testando contra pool lento ou há job switching
Solução: Testar em modo offline (sem pool) ou mudar pool
```

### Cenário B: Kernel Time é ~1350ms
```
Kernel: 50M nonces @ 1350ms = 37 MH/s
Conclusão: GPU está mesmo lenta
Causa possível: 128 threads/block é subótimo (12.5% occupancy)
Solução: Testar 256, 384, 512 threads/block
```

### Cenário C: Batch size está diminuindo automaticamente
```
Se vera batch_nonces variar: 50M → 30M → 15M → ...
Conclusão: Adaptive batching está reduzindo tamanho
Causa: Kernel > 1200ms e diminuindo
Solução: Mesma do Cenário B
```

## Próximas Ações Após Teste

**Se Kernel MH/s é 125 MH/s:**
1. Problema é pool overhead / job switching
2. Testar em modo "offline mining" (simular jobs sem pool)
3. Ou mudar para diferente pool (lower latency)

**Se Kernel MH/s é 37-50 MH/s:**
1. Problema é realmente GPU lenta
2. Testar threads_per_block = 256, 384, 512
3. Se melhora: MARCA COMO GANHO RÁPIDO
4. Se não muda: Precisa ocupancy + memoria otimization

**Se Kernel MH/s é altamente variável (100-40 MH/s):**
1. Adaptive batching está piorando
2. Kernel time varia muito
3. Possível thermal throttling do GPU?
4. Testar com `nvidia-smi -pm 1` para persistent mode

## Comandos Rápidos

```bash
# Build
cd ~/develop/rust-miner && cargo build --release

# Test with logging (requer pool configurado)
RUST_LOG=info ./target/release/rust-miner --url stratum+tcp://pool.example.com:3333 --user user 2>&1 | grep "GPU poll"

# Parse kernel hashrate
RUST_LOG=info ./target/release/rust-miner --url ... --user ... 2>&1 | grep "GPU poll done" | awk -F'=' '{print $NF}' | sort | uniq -c

# Monitor GPU (em outro terminal)
watch -n 0.1 nvidia-smi
```

---

**PRÓXIMO PASSO**: Rode o teste acima e compartilhe output com grep "GPU poll done"
