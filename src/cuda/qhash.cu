/*
 * QHash (Quantum Proof-of-Work) CUDA Kernel - OPTIMIZED
 * 
 * This kernel implements the qhash algorithm for GPU mining:
 * 1. SHA256 of block header
 * 2. Split into nibbles (4-bit values)
 * 3. Quantum circuit simulation (parameterized rotations + CNOT)
 * 4. Fixed-point conversion
 * 5. Final SHA256
 * 
 * Performance target: ~500 MH/s on GTX 1660 SUPER
 * 
 * Optimization flags applied during NVRTC compilation:
 * - -O3: Maximum optimization level
 * - --use_fast_math: Fast FP32 operations
 * - --gpu-architecture=compute_75: Turing-specific optimization
 */

// Use CUDA built-in types instead of stdint.h
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef short int16_t;

// SHA-256 constants
__constant__ uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// SHA-256 initial hash values
__constant__ uint32_t H0[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

// Cosine lookup table for angles k * (PI/32), k = 0..31
// Used to replace expensive trig calls in the quantum simulation
__constant__ float COS_PI_OVER_32[32] = {
    1.000000000f,  0.995184727f,  0.980785280f,  0.956940353f,
    0.923879504f,  0.881921291f,  0.831469595f,  0.773010433f,
    0.707106769f,  0.634393275f,  0.555570245f,  0.471396744f,
    0.382683426f,  0.290284663f,  0.195090324f,  0.098017141f,
    0.000000000f, -0.098017141f, -0.195090324f, -0.290284663f,
   -0.382683426f, -0.471396744f, -0.555570245f, -0.634393275f,
   -0.707106769f, -0.773010433f, -0.831469595f, -0.881921291f,
   -0.923879504f, -0.956940353f, -0.980785280f, -0.995184727f
};

// Helper macros for SHA-256
#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

/*
 * SHA-256 transform (single block) - optimized version
 */
__device__ void sha256_transform(const uint8_t *data, uint32_t *hash) {
    uint32_t w[64];
    uint32_t a, b, c, d, e, f, g, h;
    
    // Prepare message schedule with unrolled loop for better performance
    #pragma unroll 16
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)data[i * 4] << 24) |
               ((uint32_t)data[i * 4 + 1] << 16) |
               ((uint32_t)data[i * 4 + 2] << 8) |
               ((uint32_t)data[i * 4 + 3]);
    }
    
    // Expand message schedule
    #pragma unroll 48
    for (int i = 16; i < 64; i++) {
        w[i] = SIG1(w[i - 2]) + w[i - 7] + SIG0(w[i - 15]) + w[i - 16];
    }
    
    // Initialize working variables
    a = hash[0]; b = hash[1]; c = hash[2]; d = hash[3];
    e = hash[4]; f = hash[5]; g = hash[6]; h = hash[7];
    
    // Main loop - fully unrolled for maximum performance
    #pragma unroll 64
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + EP1(e) + CH(e, f, g) + K[i] + w[i];
        uint32_t t2 = EP0(a) + MAJ(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    
    // Add to hash
    hash[0] += a; hash[1] += b; hash[2] += c; hash[3] += d;
    hash[4] += e; hash[5] += f; hash[6] += g; hash[7] += h;
}

/*
 * SHA-256 of 80-byte block header - optimized version
 */
__device__ void sha256_80(const uint8_t *input, uint8_t *output) {
    uint32_t hash[8];
    
    // Copy H0
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        hash[i] = H0[i];
    }
    
    // First block (64 bytes of header) - optimized
    sha256_transform(input, hash);
    
    // Second block (16 bytes of header + padding) - optimized
    uint8_t padded[64];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        padded[i] = input[64 + i];
    }
    padded[16] = 0x80; // Padding bit
    #pragma unroll
    for (int i = 17; i < 62; i++) {
        padded[i] = 0;
    }
    // Length in bits (80 * 8 = 640 = 0x280)
    padded[62] = 0x02;
    padded[63] = 0x80;
    
    sha256_transform(padded, hash);
    
    // Output hash (big-endian) - optimized
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t h = hash[i];
        output[i * 4] = (h >> 24) & 0xFF;
        output[i * 4 + 1] = (h >> 16) & 0xFF;
        output[i * 4 + 2] = (h >> 8) & 0xFF;
        output[i * 4 + 3] = h & 0xFF;
    }
}

/*
 * Double SHA-256 (for final hash)
 */
__device__ void sha256d_32(const uint8_t *input, uint8_t *output) {
    uint8_t temp[32];
    uint32_t hash[8];
    uint8_t padded[64];
    
    // First SHA-256
    for (int i = 0; i < 8; i++) {
        hash[i] = H0[i];
    }
    
    for (int i = 0; i < 32; i++) {
        padded[i] = input[i];
    }
    padded[32] = 0x80;
    for (int i = 33; i < 62; i++) {
        padded[i] = 0;
    }
    padded[62] = 0x01; // 256 bits = 0x100
    padded[63] = 0x00;
    
    sha256_transform(padded, hash);
    
    // Store intermediate result
    for (int i = 0; i < 8; i++) {
        temp[i * 4] = (hash[i] >> 24) & 0xFF;
        temp[i * 4 + 1] = (hash[i] >> 16) & 0xFF;
        temp[i * 4 + 2] = (hash[i] >> 8) & 0xFF;
        temp[i * 4 + 3] = hash[i] & 0xFF;
    }
    
    // Second SHA-256
    for (int i = 0; i < 8; i++) {
        hash[i] = H0[i];
    }
    sha256_transform(temp, hash);
    
    // Output
    for (int i = 0; i < 8; i++) {
        output[i * 4] = (hash[i] >> 24) & 0xFF;
        output[i * 4 + 1] = (hash[i] >> 16) & 0xFF;
        output[i * 4 + 2] = (hash[i] >> 8) & 0xFF;
        output[i * 4 + 3] = hash[i] & 0xFF;
    }
}

/*
 * SHA-256 of exactly 64 bytes (two-block processing due to padding) - optimized
 */
__device__ void sha256_64(const uint8_t *data, uint8_t *output) {
    uint32_t hash[8];
    uint8_t pad[64];

    // Initialize hash with H0 - optimized
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        hash[i] = H0[i];
    }

    // First block: the 64 bytes of input
    sha256_transform(data, hash);

    // Second block: 0x80 + zeros + message length (512 bits) - optimized
    pad[0] = 0x80;
    #pragma unroll
    for (int i = 1; i < 62; i++) pad[i] = 0;
    pad[62] = 0x02; // 512 bits = 0x0200
    pad[63] = 0x00;

    sha256_transform(pad, hash);

    // Output final hash (big-endian) - optimized
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t h = hash[i];
        output[i * 4] = (h >> 24) & 0xFF;
        output[i * 4 + 1] = (h >> 16) & 0xFF;
        output[i * 4 + 2] = (h >> 8) & 0xFF;
        output[i * 4 + 3] = h & 0xFF;
    }
}

/*
 * Split bytes into 4-bit nibbles
 */
__device__ void split_nibbles(const uint8_t *data, uint8_t *nibbles, int len) {
    #pragma unroll
    for (int i = 0; i < len; i++) {
        nibbles[i * 2] = (data[i] >> 4) & 0x0F;     // High nibble
        nibbles[i * 2 + 1] = data[i] & 0x0F;        // Low nibble
    }
}

/*
 * Quantum circuit simulation (optimized for GPU with shared memory)
 */
__device__ void quantum_simulation(const uint8_t *nibbles, int nibbles_len, uint32_t ntime, float *expectations) {
    const int N_QUBITS = 16;
    const int N_LAYERS = 2;
    
    // Initialize expectations in shared memory for better performance
    __shared__ float shared_expectations[256]; // 256 threads max
    
    int tid = threadIdx.x;
    if (tid < N_QUBITS) {
        shared_expectations[tid] = 0.0f;
    }
    __syncthreads();
    
    // Protocol upgrade flag
    int upgrade = (ntime >= 1758762000) ? 1 : 0;
    
    // Apply layers with optimized memory access
    #pragma unroll
    for (int l = 0; l < N_LAYERS; l++) {
        // Rotation gates - compute all at once for better ILP
        if (tid < N_QUBITS) {
            uint8_t ry_nibble = nibbles[(2 * l * N_QUBITS + tid) % nibbles_len];
            uint8_t rz_nibble = nibbles[((2 * l + 1) * N_QUBITS + tid) % nibbles_len];

            // Optimized angle calculation using lookup table
            int idx_ry = ((ry_nibble << 1) + upgrade) & 31;
            int idx_rz = ((rz_nibble << 1) + upgrade) & 31;
            float z_exp = COS_PI_OVER_32[idx_ry] * COS_PI_OVER_32[idx_rz];
            shared_expectations[tid] += z_exp;
        }
        __syncthreads();
        
        // CNOT entanglement - optimized with warp-level operations
        if (tid < N_QUBITS - 1) {
            float coupling = 0.1f;
            float exp_i = shared_expectations[tid];
            float exp_ip1 = shared_expectations[tid + 1];
            
            // Use fused multiply-add for better performance
            shared_expectations[tid] = fmaf(exp_i, 1.0f - coupling, exp_ip1 * coupling);
            shared_expectations[tid + 1] = fmaf(exp_ip1, 1.0f - coupling, exp_i * coupling);
        }
        __syncthreads();
    }
    
    // Copy back to global memory and normalize
    if (tid < N_QUBITS) {
        expectations[tid] = tanhf(shared_expectations[tid]);
    }
}

/*
 * Convert float to fixed-point int16 - optimized version
 */
__device__ __forceinline__ int16_t to_fixed_point(float value) {
    // Clamp to [-1.0, 1.0] and convert - optimized with intrinsics
    return (int16_t)(fmaxf(fminf(value, 1.0f), -1.0f) * 32768.0f);
}

/*
 * Main QHash mining kernel - optimized version
 * 
 * Each thread processes one nonce
 */
extern "C" __global__ void qhash_mine(
    const uint8_t *block_header,    // 76 bytes (without nonce)
    uint32_t ntime,                  // Network time for protocol upgrade
    uint32_t start_nonce,            // Starting nonce
    const uint8_t *target,           // 32-byte target (big-endian)
    uint32_t *solution,              // Output: found nonce (or 0xFFFFFFFF if none)
    uint8_t *found_hash,             // Output: 32-byte hash when solution found
    uint32_t num_threads             // Total threads
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid >= num_threads) return;
    
    uint32_t nonce = start_nonce + gid;
    
    // Load block header into shared memory (once per block)
    __shared__ uint8_t shared_header[76];
    __shared__ uint8_t shared_target[32];
    
    // Load header and target to shared memory cooperatively
    if (threadIdx.x < 76) {
        shared_header[threadIdx.x] = block_header[threadIdx.x];
    }
    if (threadIdx.x < 32) {
        shared_target[threadIdx.x] = target[threadIdx.x];
    }
    __syncthreads();
    
    // Build complete header with nonce in registers for speed
    uint8_t header[80];
    #pragma unroll
    for (int i = 0; i < 76; i++) {
        header[i] = shared_header[i];
    }
    // Append nonce (little-endian) - optimized
    uint32_t nonce_le = nonce;
    header[76] = nonce_le & 0xFF;
    header[77] = (nonce_le >> 8) & 0xFF;
    header[78] = (nonce_le >> 16) & 0xFF;
    header[79] = (nonce_le >> 24) & 0xFF;
    
    // Step 1: SHA256 of header - optimized
    uint8_t hash1[32];
    sha256_80(header, hash1);
    
    // Step 2: Split into nibbles - use registers
    uint8_t nibbles[64];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        nibbles[i * 2] = (hash1[i] >> 4) & 0x0F;     // High nibble
        nibbles[i * 2 + 1] = hash1[i] & 0x0F;        // Low nibble
    }
    
    // Step 3: Quantum simulation - optimized with registers
    float expectations[16];
    const int N_QUBITS = 16;
    const int N_LAYERS = 2;
    
    // Initialize expectations
    #pragma unroll
    for (int i = 0; i < N_QUBITS; i++) {
        expectations[i] = 0.0f;
    }
    
    // Protocol upgrade flag
    int upgrade = (ntime >= 1758762000) ? 1 : 0;
    
    // Apply layers with optimized operations
    #pragma unroll
    for (int l = 0; l < N_LAYERS; l++) {
        // Rotation gates - optimized
        #pragma unroll
        for (int i = 0; i < N_QUBITS; i++) {
            uint8_t ry_nibble = nibbles[(2 * l * N_QUBITS + i) % 64];
            uint8_t rz_nibble = nibbles[((2 * l + 1) * N_QUBITS + i) % 64];

            int idx_ry = ((ry_nibble << 1) + upgrade) & 31;
            int idx_rz = ((rz_nibble << 1) + upgrade) & 31;
            float z_exp = COS_PI_OVER_32[idx_ry] * COS_PI_OVER_32[idx_rz];
            expectations[i] += z_exp;
        }
        
        // CNOT entanglement - optimized
        #pragma unroll
        for (int i = 0; i < N_QUBITS - 1; i++) {
            float coupling = 0.1f;
            float temp = expectations[i] * (1.0f - coupling) + expectations[i + 1] * coupling;
            expectations[i + 1] = expectations[i + 1] * (1.0f - coupling) + expectations[i] * coupling;
            expectations[i] = temp;
        }
    }
    
    // Normalize with tanh - optimized
    #pragma unroll
    for (int i = 0; i < N_QUBITS; i++) {
        expectations[i] = tanhf(expectations[i]);
    }
    
    // Step 4: Convert to fixed-point - optimized
    uint8_t quantum_output[32];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int16_t fp = to_fixed_point(expectations[i]);
        quantum_output[i * 2] = fp & 0xFF;
        quantum_output[i * 2 + 1] = (fp >> 8) & 0xFF;
    }
    
    // Step 5: Final SHA256 - optimized concatenation
    uint8_t final_input[64];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        final_input[i] = hash1[i];
        final_input[i + 32] = quantum_output[i];
    }
    
    uint8_t final_hash[32];
    sha256_64(final_input, final_hash);

    // Protocol upgrade: reject blocks with too many zero bytes
    int zero_count = 0;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        zero_count += (quantum_output[i] == 0) ? 1 : 0;
    }
    
    const int total_bytes = 32;
    bool should_reject = (zero_count == total_bytes && ntime >= 1753105444) ||
                        (zero_count >= (total_bytes * 3 / 4) && ntime >= 1753305380) ||
                        (zero_count >= (total_bytes / 4) && ntime >= 1754220531);
    
    // Early exit for rejected blocks
    if (should_reject) {
        return;
    }
    
    // Check if hash meets target (big-endian comparison) - optimized
    bool meets_target = true;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        if (final_hash[i] < shared_target[i]) {
            break; // hash < target, valid!
        } else if (final_hash[i] > shared_target[i]) {
            meets_target = false;
            break;
        }
    }
    
    // If found, store nonce and hash atomically
    if (meets_target) {
        // Use atomicCAS to ensure only first thread writes
        uint32_t old = atomicCAS(solution, 0xFFFFFFFF, nonce);
        if (old == 0xFFFFFFFF) {
            // This thread won the race - write the hash
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                found_hash[i] = final_hash[i];
            }
        }
    }
}
