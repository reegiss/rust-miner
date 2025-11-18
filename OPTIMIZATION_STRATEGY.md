# Estratégia de Otimização de Hashrate: SHA256 em PTX Assembly

## 📊 Estado Atual
- **Hashrate Atual**: ~37 MH/s (GTX 1660 SUPER)
- **Target**: ~500 MH/s (13.5x de melhoria)
- **Gargalo Principal**: SHA256 C++ em loop (sha256_transform)
- **Limite Atingido**: Unroll loops em C++ não é suficiente

## 🎯 Objetivo
Substituir o `sha256_transform` em C++ por **PTX inline assembly** para:
1. Reduzir latência de memória (usar registros ao invés de stack)
2. Explorar paralelismo de warp com instrução `shfl.sync`
3. Usar `add.cc` (add-with-carry) para operações otimizadas
4. Eliminar overhead de branches no loop de 64 rounds

## 📚 Análise de Código Encontrado

### Fonte: ccminer (tpruvot)
- `scrypt/sha256.cu`: Implementação CUDA com RNDr macro otimizado
- `scrypt.cpp`: Versão vectorizada com SSE/AVX para CPU
- Padrão: RNDr(S, W, i) expande para operações com `add`, `xor`, `rotr`

### Macro Crítico Encontrado
```c
#define RND(a, b, c, d, e, f, g, h, k) \
	do { \
		t0 = h + S1(e) + Ch(e, f, g) + k; \
		t1 = S0(a) + Maj(a, b, c); \
		d += t0; \
		h  = t0 + t1; \
	} while (0)
```

Isto é executado 64 vezes por bloco. **Este é o gargalo.**

## 🔧 Estratégia de Implementação (PTX)

### Fase 1: Extrair SHA256 para PTX Inline Assembly
1. Mover o loop de 64 RNDr para `asm volatile` com registros
2. Usar PTX `add.cc` e `addc` para operações com carry
3. Manter W[64] em registros (ou cache local otimizado)

### Fase 2: Paralelismo de Warp
1. Usar `__shfl_sync` para partilhar valores de S[i] entre threads num warp
2. Executar SHA256 em paralelo em múltiplos nonces

### Fase 3: Otimizações Secundárias
1. Precompute K[64] no kernel (em `__constant__`)
2. Loop unrolling manual em PTX (cada round é uma sequência de PTX)
3. Reduzir memory pressure com coalescing otimizado

## 📝 Próximos Passos

### [TAREFA 1.1] Pesquisa de Implementação Existente
- [ ] Procurar `wildrig-multi` (open-source, também usa CUDA QHash)
- [ ] Procurar repositórios de mining CUDA especializados
- [ ] Validar que PTX assembly é a abordagem correta

### [TAREFA 1.2] Prototipagem de PTX Inline Assembly
- [ ] Criar função `__device__ void sha256_transform_asm(uint32_t *state, uint32_t *block)`
- [ ] Implementar 2-3 rounds em PTX como prova de conceito
- [ ] Medir redução de latência vs. versão C++

### [TAREFA 1.3] Deploy Completo
- [ ] Portar todos os 64 rounds para PTX
- [ ] Integrar com kernel QHash existente
- [ ] Testar hashrate completo

## ⚠️ Considerações de Risco

1. **Compatibilidade de GPU**: PTX é versionado. Precisamos de SM 3.5+ (GTX 750 Ti, 9xx, 10xx+)
2. **Debugging**: PTX é complexo; erros podem causar silent corruption
3. **Maintenance**: Código PT é mais difícil de manter que C++
4. **Fallback**: Manter versão C++ como fallback se PTX compilar com erros

## 📦 Implementação Alternativa (Mais Segura)

Se PTX assembly for demasiado complexo, investigar:
1. **ONNX Runtime** ou **TVM** para geração de código otimizado
2. **Blocos de C++ com `#pragma unroll`** e `-O3 optimization flags`
3. **Splittar SHA256 em múltiplos kernels** para reduzir latência

---

**Próximo**: [PESQUISAR: "wildrig-multi cuda qhash"] para validar arquitetura
