# Otimizações Implementadas - Job Switching Overhead

## Problema Identificado

**Antes**: 37 MH/s observado vs 325 MH/s kernel (88.6% perda)

**Causa**: Código estava checando por novo job a cada 10 iterações (`if iterations % 10 == 0`) **DURANTE** o processamento de batches.

Isto causava:
1. Interrupção imediata de batch em andamento
2. Nonces perdidos (GPU continuava, mas Rust já contava como "done")
3. Overhead de job switching frequente

## Otimizações Implementadas

### 1. ✅ Job Switch Depois, Não Durante
**Antes**:
```rust
// Check for new job every 10 iterations (DURANTE batch processing)
if iterations % 10 == 0 && stratum_client.has_pending_job().await {
    println!("   {} Switching to new job", "🔄".yellow());
    break 'gpu_mining;
}
```

**Depois**:
```rust
// Check for new job AFTER batch completes (ao fim de cada batch)
if stratum_client.has_pending_job().await {
    println!("   {} Switching to new job (after batch)", "🔄".yellow());
    break 'gpu_mining;
}
```

**Impacto**: Deixa cada batch rodar até o fim. Reduz overhead de context switching.

### 2. ✅ Adaptive Batching Desabilitado (Temporário)
**Antes**:
```rust
// Target window ~700-900ms, ajusta chunk_size dinamicamente
if ms < 400 && chunk_size < 150_000_000 {
    chunk_size = (chunk_size as f64 * 1.25) as u32;  // +25%
} else if ms > 1200 && chunk_size > 5_000_000 {
    chunk_size = (chunk_size as f64 * 0.80) as u32;  // -20%
}
```

**Depois**:
```rust
// DISABLED FOR NOW: Testing with fixed 50M nonces
// (deixa comentado, pode re-habilitar depois)
```

**Impacto**: Mantém batch size consistente em 50M nonces. Elimina variabilidade.

**Teoria**: Adaptive batching estava reduzindo para 13-15M nonces (o que explica ~37 MH/s).

### 3. ✅ Menos Prints de Stats (menos overhead)
**Antes**:
```rust
if iterations % 25 == 0 {
    // Print stats (chama async function)
}
```

**Depois**:
```rust
if iterations % 100 == 0 {
    // Print stats com menos frequência
}
```

**Impacto**: Reduz overhead de logging. Cada print chama `print_wildrig_stats` que tem overhead.

## Esperado Ganho

Com as otimizações acima:
- Overhead de job switching: reduzido
- Batch size: fixo em 50M nonces
- Logging overhead: reduzido

**Esperado**: 37 MH/s → **200-300 MH/s** (perto do kernel 325 MH/s)

## Como Testar

```bash
# Build
cargo build --release

# Run com pool (qualquer pool stratum)
./target/release/rust-miner --url stratum+tcp://pool.example.com:3333 --user user

# Ver logs
RUST_LOG=info ./target/release/rust-miner ... 2>&1 | grep -E "GPU:|Switching to new job|MH/s"
```

## Métricas para Validar

Esperado ver:
1. **"GPU: XXX.XX MH/s"** no log deve estar próximo a 325 MH/s
2. **"Switching to new job"** menos frequente (apenas quando novo job chega)
3. **Batch_nonces** consistente em 50M

Se ver:
- GPU MH/s = 325: ✅ kernel está perfeito
- Mas total < 100 MH/s: ⚠️ ainda há overhead (pool latency ou outro)

## Próximas Ações Se Ainda Lento

Se mesmo após otimizações ainda está <100 MH/s:

1. **Re-habilitar adaptive batching** (talvez 50M é pequeno)
   - Tentar 100M, 200M, 500M nonces por batch

2. **Profiling de pool latency**
   - Ver quanto tempo leva stratum client responder

3. **Múltiplos threads de mining**
   - Executar N GPU kernels em paralelo?

4. **NVRTC Compilation flags**
   - Adicionar `-O3`, `-use_fast_math`, etc.

## Commits a Fazer

```bash
git add src/main.rs
git commit -m "Otimização: Job switch após batch, não durante

- Remover verificação de novo job durante batch (iterations % 10)
- Adicionar verificação DEPOIS de cada batch completar
- Desabilitar adaptive batching (testar com 50M fixo)
- Reduzir frequência de logging (iterations % 100)

Impacto esperado: Reduzir overhead de job switching ~80%
Resultado: 37 MH/s → target 200-325 MH/s"
```

---

**Status**: Otimizações implementadas. Pronto para testar!
