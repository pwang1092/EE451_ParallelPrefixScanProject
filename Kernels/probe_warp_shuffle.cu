/**
 * probe_warp_shuffle.cu — Find the maximum L per block for WarpShuffle at each D.
 *
 * Increments L (power-of-2) until a CUDA error occurs, reporting the last
 * successful L and the shared memory usage at that point.
 *
 * Build:
 *   nvcc -O2 -std=c++17 -arch=sm_80 -DD=16  --maxrregcount=64 -o probe_ws_D16  probe_warp_shuffle.cu
 *   nvcc -O2 -std=c++17 -arch=sm_80 -DD=64  --maxrregcount=64 -o probe_ws_D64  probe_warp_shuffle.cu
 *   nvcc -O2 -std=c++17 -arch=sm_80 -DD=256 --maxrregcount=64 -o probe_ws_D256 probe_warp_shuffle.cu
 *   nvcc -O2 -std=c++17 -arch=sm_80 -DD=512 --maxrregcount=64 -o probe_ws_D512 probe_warp_shuffle.cu
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "warp_shuffle.cu"  // pulls in common.cuh then warp_shuffle.cuh

__global__ void warp_shuffle_kernel(Element* d_in, Element* d_out, int L) {
    __shared__ Element shared_data[BLOCK_SIZE];
    __shared__ Element block_total;

    int tid = threadIdx.x;
    if (tid < L) shared_data[tid] = d_in[tid];
    __syncthreads();

    WarpShuffle::block_scan(shared_data, L, &block_total);

    if (tid < L) d_out[tid] = shared_data[tid];
}

int main() {
    printf("WarpShuffle shared memory probe  D=%d  BLOCK_SIZE=%d\n\n", D, BLOCK_SIZE);
    printf("  Shared memory per block = (BLOCK_SIZE + BLOCK_SIZE/32 + 1) * 2 * D * 4 bytes\n");
    printf("  = (%d + %d + 1) * %d * 4 = %zu bytes = %.1f KB\n\n",
           BLOCK_SIZE, BLOCK_SIZE/32,
           2*D,
           (size_t)(BLOCK_SIZE + BLOCK_SIZE/32 + 1) * 2 * D * 4,
           (BLOCK_SIZE + BLOCK_SIZE/32 + 1) * 2.0 * D * 4 / 1024.0);

    printf("%-8s %-12s %-10s %-s\n", "L", "shmem_KB", "threads", "result");
    printf("%.50s\n", "--------------------------------------------------");

    int last_pass = -1;
    // probe from 32 up to 1024 in powers of 2
    for (int L = 32; L <= 1024; L *= 2) {
        if (L > BLOCK_SIZE) {
            printf("%-8d %-12s %-10s SKIP (exceeds BLOCK_SIZE=%d)\n",
                   L, "-", "-", BLOCK_SIZE);
            continue;
        }

        size_t shmem = (size_t)(BLOCK_SIZE + BLOCK_SIZE/32 + 1) * sizeof(Element);
        double shmem_kb = shmem / 1024.0;

        std::vector<Element> h_in(L), h_out(L);
        for (int t = 0; t < L; t++)
            for (int d = 0; d < D; d++) {
                h_in[t].a[d] = 0.9f;
                h_in[t].b[d] = 1.0f;
            }

        Element *d_in, *d_out;
        cudaMalloc(&d_in,  L * sizeof(Element));
        cudaMalloc(&d_out, L * sizeof(Element));
        cudaMemcpy(d_in, h_in.data(), L * sizeof(Element), cudaMemcpyHostToDevice);

        warp_shuffle_kernel<<<1, BLOCK_SIZE>>>(d_in, d_out, L);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();

        printf("%-8d %-12.1f %-10d %s\n",
               L, shmem_kb, BLOCK_SIZE,
               err == cudaSuccess ? "PASS" : cudaGetErrorString(err));

        if (err == cudaSuccess) last_pass = L;

        cudaFree(d_in);
        cudaFree(d_out);

        if (err != cudaSuccess) break;
    }

    printf("\nMax L per block for D=%d: %d\n", D, last_pass);
    return 0;
}
