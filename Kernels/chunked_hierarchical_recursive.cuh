#ifndef CHUNKED_HIERARCHICAL_RECURSIVE_CUH
#define CHUNKED_HIERARCHICAL_RECURSIVE_CUH

#include "common.cuh"
#include "hillis_steele.cuh"
#include "blelloch.cuh"
#include "warp_shuffle.cuh"

// ---------------------------------------------------------------------------
// CHUNK_SIZE: minimum BLOCK_SIZE across all three kernels for given D,
// ensuring every ScanImpl fits within its shared memory budget.
//   D=16:  min(512, 512, 512) = 512
//   D=64:  min(256, 128, 128) = 128
//   D=256: min(64,  32,  32)  = 32
//   D=512: min(32,  16,  16)  = 16
// ---------------------------------------------------------------------------
#ifdef CHUNK_SIZE
#undef CHUNK_SIZE
#endif

#if D <= 16
#define CHUNK_SIZE 512
#elif D <= 64
#define CHUNK_SIZE 128
#elif D <= 256
#define CHUNK_SIZE 32
#else
#define CHUNK_SIZE 16
#endif

// ---------------------------------------------------------------------------
// Phase 1: each block scans its own chunk independently.
// Uses dynamic shared memory to avoid compile-time ptxas limit.
// ---------------------------------------------------------------------------
template <typename ScanImpl>
__global__ void phase1_local_scan(
    const Element* __restrict__ input,
          Element* __restrict__ output,
          Element* __restrict__ block_totals,
    int L)
{
    extern __shared__ Element shared_data[];

    int block_start = blockIdx.x * CHUNK_SIZE;
    int tid         = threadIdx.x;
    int global_idx  = block_start + tid;

    shared_data[tid] = (global_idx < L) ? input[global_idx] : identity();
    __syncthreads();

    Element total;
    ScanImpl::block_scan(shared_data, CHUNK_SIZE, &total);
    __syncthreads();

    if (global_idx < L)
        output[global_idx] = shared_data[tid];

    if (tid == 0)
        block_totals[blockIdx.x] = total;
}

// ---------------------------------------------------------------------------
// Phase 2 (single-block): scan block totals when they fit in one block.
// Uses dynamic shared memory to avoid compile-time ptxas limit.
// ---------------------------------------------------------------------------
template <typename ScanImpl>
__global__ void phase2_scan_totals(
    Element* __restrict__ block_totals,
    int num_blocks)
{
    extern __shared__ Element shared_data[];

    int tid = threadIdx.x;
    shared_data[tid] = (tid < num_blocks) ? block_totals[tid] : identity();
    __syncthreads();

    Element total;
    ScanImpl::block_scan(shared_data, num_blocks, &total);
    __syncthreads();

    if (tid < num_blocks)
        block_totals[tid] = shared_data[tid];
}

// ---------------------------------------------------------------------------
// Phase 3: add preceding block's scanned total into each element.
// Block 0 is already globally correct.
// ---------------------------------------------------------------------------
__global__ void phase3_propagate(
          Element* __restrict__ output,
    const Element* __restrict__ block_totals,
    int L)
{
    if (blockIdx.x == 0) return;

    int global_idx = blockIdx.x * CHUNK_SIZE + threadIdx.x;
    if (global_idx >= L) return;

    output[global_idx] = combine(block_totals[blockIdx.x - 1], output[global_idx]);
}

// ---------------------------------------------------------------------------
// chunked_scan — recursive 3-phase scan supporting arbitrary L.
// Phase 2 recurses if num_blocks > CHUNK_SIZE.
// ---------------------------------------------------------------------------
template <typename ScanImpl>
void chunked_scan(Element* d_input, Element* d_output, int L)
{
    if (L <= 0) return;

    int num_blocks = (L + CHUNK_SIZE - 1) / CHUNK_SIZE;
    size_t shmem = CHUNK_SIZE * sizeof(Element);

    Element* d_block_totals;
    cudaMalloc(&d_block_totals, num_blocks * sizeof(Element));

    // Phase 1: local scan per block
    cudaFuncSetAttribute(phase1_local_scan<ScanImpl>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    phase1_local_scan<ScanImpl><<<num_blocks, CHUNK_SIZE, shmem>>>(
        d_input, d_output, d_block_totals, L);

    // Phase 2: scan block totals — recursive if too many blocks
    if (num_blocks <= CHUNK_SIZE) {
        cudaFuncSetAttribute(phase2_scan_totals<ScanImpl>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
        phase2_scan_totals<ScanImpl><<<1, CHUNK_SIZE, shmem>>>(
            d_block_totals, num_blocks);
    } else {
        Element* d_scanned_totals;
        cudaMalloc(&d_scanned_totals, num_blocks * sizeof(Element));
        chunked_scan<ScanImpl>(d_block_totals, d_scanned_totals, num_blocks);
        cudaMemcpy(d_block_totals, d_scanned_totals,
                   num_blocks * sizeof(Element), cudaMemcpyDeviceToDevice);
        cudaFree(d_scanned_totals);
    }

    // Phase 3: propagate block prefixes
    phase3_propagate<<<num_blocks, CHUNK_SIZE>>>(d_output, d_block_totals, L);

    cudaFree(d_block_totals);
}

#endif // CHUNKED_HIERARCHICAL_RECURSIVE_CUH
