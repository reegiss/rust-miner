# [PESQUISA 1] Resultados - SHA256 PTX Assembly para Mineração CUDA

## Status
**Pesquisa Concluída**: Encontramos referências em ccminer (tpruvot) que implementam SHA256 em CUDA, mas principalmente usando C++ com macros e `#pragma unroll`, não PTX inline assembly.

## Descobertas Críticas

### 1. ccminer (tpruvot/ccminer) - Referência Principal
- **Repo**: github.com/tpruvot/ccminer
- **Arquivos SHA256**: 
  - `scrypt/sha256.cu` (linha 158-183): `__device__ void cuda_sha256_transform()`
  - `lbry/cuda_sha256_lbry.cu` (linha 178-195): sha256 com asm volatile para byte permutations
  - `sha256/cuda_sha256d.cu`: SHA256d mining kernel
  - `sha256/cuda_sha256t.cu`: SHA256 3x triple hash

### 2. Padrões Encontrados em ccminer

#### A. Loop Unrolling C++ (NÃO Assembly)
```cuda
// From scrypt/sha256.cu
#define RNDr(S, W, i) \
	RND(S[(64 - i) % 8], S[(65 - i) % 8], \
		S[(66 - i) % 8], S[(67 - i) % 8], \
		S[(68 - i) % 8], S[(69 - i) % 8], \
		S[(70 - i) % 8], S[(71 - i) % 8], \
		W[i] + sha256_k[i])

// 64 rounds fully expanded as macros:
RNDr(S, W,  0); RNDr(S, W,  1); RNDr(S, W,  2); ... RNDr(S, W, 63);
```

#### B. PTX Inline Assembly - Encontrado em lbry/cuda_sha256_lbry.cu
```cuda
// Line 187-192: Byte permutation with PTX assembly
__device__ __forceinline__ uint2 vectorizeswap(uint64_t v)
{
	uint2 result;
	asm("mov.b64 {%0,%1},%2; // vectorizeswap \n\t"
		: "=r"(result.y), "=r"(result.x) : "l"(v));
	return result;
}
```

**Padrão**: `asm("PTX_INSTRUCTION" : output_constraints : input_constraints);`

### 3. Estrutura de SHA256 em CUDA (de ccminer)

```cuda
__device__ void cuda_sha256_transform(uint32_t *state, const uint32_t *block)
{
	uint32_t W[64];      // Message schedule
	uint32_t S[8];       // State variables (a-h)
	uint32_t t0, t1;     // Temporaries
	
	// Initialize
	mycpy32(S, state);   // Copy state
	mycpy16(W, block);   // Copy first 16 words
	
	// Expand message schedule (16 → 64)
	#pragma unroll 2
	for (i = 16; i < 64; i += 2) {
		W[i]   = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
		W[i+1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15];
	}
	
	// 64 rounds (fully unrolled via RNDr macros)
	RNDr(S, W,  0); RNDr(S, W,  1); ... RNDr(S, W, 63);
	
	// Add to state
	#pragma unroll 8
	for (i = 0; i < 8; i++)
		state[i] += S[i];
}
```

### 4. PTX Inline Assembly - Oportunidades

Do que encontrei no ccminer:

**Operações viáveis com asm volatile:**
- Rotações de 32-bit: `shf.r.wrap.b32`, `rol.b32`, `ror.b32`
- Carry operations: `add.cc`, `addc` (para operações com borrow/carry)
- Byte permutations: `mov.b64`, `__byte_perm`
- Logical: `lop3.b32` (fused 3-input logic)
- Memory: Coalesced loads/stores already handled by NVCC

## 🚨 VERDADE INCÔMODA

ccminer **NÃO usa PTX inline assembly para o core SHA256 loop**. Usa apenas:
1. **C++ macros** com `#pragma unroll 64` - deixa para compiler otimizar
2. **PTX asm** apenas para small helper functions (byte permutation, register swaps)
3. **Razão**: 
   - NVRTC compiler é excelente em otimizar código C++
   - PTX assembly manual é difícil de debugar e manter
   - Compiler já faz scheduling e register allocation automático

## ⚠️ Implicação para rust-miner

**Contra-evidência**: Se ccminer (o miner de referência) não usa PTX inline assembly para SHA256 core, talvez a estratégia de "500 MH/s via PTX" seja baseada em falsa premissa.

**Possibilidade 1**: ccminer usa kernel kernels otimizadas mas em C++ (não asm)
**Possibilidade 2**: Diferenças entre algoritmos - ccminer é para Ethash/Scrypt, não QHash
**Possibilidade 3**: PTX inline faz mais sentido para loops paralelos que são hard to optimize

## Próximos Passos

### [OPÇÃO A] Pesquisar Mineradores de Algoritmos Blockchain Específicos
- Que usam SHA256 DUPLO como core (similar a rust-miner)
- Buscar: "bitcoin", "litecoin", "dogecoin" CUDA miners

### [OPÇÃO B] Pesquisar CUDA Optimization Papers
- Nvidia best practices para SHA256 em CUDA
- Comparison: C++ with pragmas vs. inline PTX assembly

### [OPÇÃO C] Procurar Implementações de Cryptographic Hashing
- CUF (CUDA Unified Framework)
- NVIDIA NVLabs repositories
- Research papers on GPU sha256

## Recomendação Imediata

Com base nas descobertas:
1. **NÃO há evidence clara que PTX inline assembly para SHA256 dá 13.5x**
2. **ccminer usa C++ + compiler optimization**
3. **Ganho de 37 → 500 MH/s pode vir de:**
   - Algoritmo completamente diferente (não QHash)
   - Memory hierarchy optimization
   - Better kernel launch configuration
   - Compilation flags (`-O3`, `-Xptxas`, etc)
   - Diferentes assumptions (GPU type, VRAM, etc)

---

**Conclusão**: A "estratégia 500 MH/s" pode ser baseada em wrong assumptions sobre PTX assembly sendo a solução. Recomendo validar se essa taxa é atingível para QHash especificamente, em GTX 1660 SUPER.
