#ifndef CHUNKED_HIERARCHICAL_RECURSIVE_CUH
#define CHUNKED_HIERARCHICAL_RECURSIVE_CUH

#include "common.cuh"
#include "hillis_steele.cuh"
#include "blelloch.cuh"
#include "warp_shuffle.cuh"

// ---------------------------------------------------------------------------
// CHUNK_SIZE: minimum BLOCK_SIZE across all three kernels for given D.
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
// shmem_elements<ScanImpl>: returns how many Element slots the kernel needs.
// HillisSteele needs 2*CHUNK_SIZE (double-buffer).
// Blelloch needs conflict-free padding because it accesses shared_data[phys(i)].
// ---------------------------------------------------------------------------
template <typename ScanImpl>
constexpr size_t shmem_elements() { return CHUNK_SIZE; }

template <>
constexpr size_t shmem_elements<HillisSteele>() { return 2 * CHUNK_SIZE; }

constexpr size_t blelloch_padding(size_t i) {
    // Must match CONFLICT_FREE_OFFSET(i) in blelloch.cu.
    return (i >> 5) + (i >> 10);
}

template <>
constexpr size_t shmem_elements<Blelloch>() {
    return CHUNK_SIZE + blelloch_padding(CHUNK_SIZE - 1);
}

// ---------------------------------------------------------------------------
// Phase 1: each block scans its own chunk independently.
// Uses dynamic shared memory. HillisSteele callers pass 2*CHUNK_SIZE elements.
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

    __shared__ Element s_total;  // shared so any thread can write, thread 0 reads
    ScanImpl::block_scan(shared_data, CHUNK_SIZE, &s_total);
    __syncthreads();

    if (global_idx < L)
        output[global_idx] = shared_data[tid];

    // s_total is in shared memory — valid for all threads regardless of which
    // thread inside block_scan wrote to *block_total (thread 0 for Blelloch,
    // thread n-1 for WarpShuffle and HillisSteele)
    if (tid == 0)
        block_totals[blockIdx.x] = s_total;
}

// ---------------------------------------------------------------------------
// Phase 2 (single-block): scan block totals when they fit in one block.
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
// ---------------------------------------------------------------------------
template <typename ScanImpl>
void chunked_scan(Element* d_input, Element* d_output, int L)
{
    if (L <= 0) return;

    int num_blocks = (L + CHUNK_SIZE - 1) / CHUNK_SIZE;
    // HillisSteele needs 2*CHUNK_SIZE for double-buffer; others need CHUNK_SIZE
    size_t shmem = shmem_elements<ScanImpl>() * sizeof(Element);

    Element* d_block_totals;
    cudaMalloc(&d_block_totals, num_blocks * sizeof(Element));

    // Phase 1
    cudaFuncSetAttribute(phase1_local_scan<ScanImpl>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    phase1_local_scan<ScanImpl><<<num_blocks, CHUNK_SIZE, shmem>>>(
        d_input, d_output, d_block_totals, L);

    // Phase 2 — recurse if num_blocks > CHUNK_SIZE
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

    // Phase 3
    phase3_propagate<<<num_blocks, CHUNK_SIZE>>>(d_output, d_block_totals, L);

    cudaFree(d_block_totals);
}

#endif // CHUNKED_HIERARCHICAL_RECURSIVE_CUH
