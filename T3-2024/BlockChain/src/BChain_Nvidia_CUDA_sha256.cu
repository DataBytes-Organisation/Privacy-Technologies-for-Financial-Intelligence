#include <stdio.h>
#include <stdint.h>
#include <cuda.h>

__device__ __constant__ uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// SHA-256 helper functions
__device__ uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

__device__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ uint32_t sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

__device__ uint32_t sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

__device__ uint32_t gamma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

__device__ uint32_t gamma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

__device__ void sha256_transform(const uint8_t *chunk, uint32_t *hash) {
    uint32_t w[64];
    uint32_t a, b, c, d, e, f, g, h;

    // Prepare the message schedule
    for (int i = 0; i < 16; i++) {
        w[i] = (chunk[i * 4] << 24) | (chunk[i * 4 + 1] << 16) | (chunk[i * 4 + 2] << 8) | chunk[i * 4 + 3];
    }
    for (int i = 16; i < 64; i++) {
        w[i] = gamma1(w[i - 2]) + w[i - 7] + gamma0(w[i - 15]) + w[i - 16];
    }

    // Initialize working variables
    a = hash[0];
    b = hash[1];
    c = hash[2];
    d = hash[3];
    e = hash[4];
    f = hash[5];
    g = hash[6];
    h = hash[7];

    // Main computation loop
    for (int i = 0; i < 64; i++) {
        uint32_t temp1 = h + sigma1(e) + ch(e, f, g) + k[i] + w[i];
        uint32_t temp2 = sigma0(a) + maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    // Update the hash values
    hash[0] += a;
    hash[1] += b;
    hash[2] += c;
    hash[3] += d;
    hash[4] += e;
    hash[5] += f;
    hash[6] += g;
    hash[7] += h;
}

__global__ void sha256_kernel(const uint8_t *input, uint8_t *output, int num_chunks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_chunks) {
        uint32_t hash[8] = {
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
        };

        // Compute SHA-256 hash for this chunk
        sha256_transform(&input[idx * 64], hash);

        // Store the result in global memory
        for (int i = 0; i < 8; i++) {
            output[idx * 32 + i * 4] = (hash[i] >> 24) & 0xff;
            output[idx * 32 + i * 4 + 1] = (hash[i] >> 16) & 0xff;
            output[idx * 32 + i * 4 + 2] = (hash[i] >> 8) & 0xff;
            output[idx * 32 + i * 4 + 3] = hash[i] & 0xff;
        }
    }
}

int main() {
    const int num_chunks = 256;  // Number of 64-byte chunks
    size_t input_size = num_chunks * 64;  // Total input size
    size_t output_size = num_chunks * 32; // 32-byte hash per chunk

    uint8_t *h_input, *h_output;
    uint8_t *d_input, *d_output;

    // Allocate host memory
    h_input = (uint8_t *)malloc(input_size);
    h_output = (uint8_t *)malloc(output_size);

    // Fill input with dummy data
    for (int i = 0; i < input_size; i++) h_input[i] = i % 256;

    // Allocate device memory
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);

    // Copy input to device
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    int blocks = (num_chunks + threads - 1) / threads;
    sha256_kernel<<<blocks, threads>>>(d_input, d_output, num_chunks);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

    // Print the first few hashes
    for (int i = 0; i < 5; i++) {
        printf("Hash %d: ", i);
        for (int j = 0; j < 32; j++) printf("%02x", h_output[i * 32 + j]);
        printf("\n");
    }

    // Clean up
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
