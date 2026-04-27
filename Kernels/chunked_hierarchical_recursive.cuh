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
    return (i >> 5) + (i >> 10);
}

template <>
constexpr size_t shmem_elements<Blelloch>() {
    return CHUNK_SIZE + blelloch_padding(CHUNK_SIZE - 1);
}

// ---------------------------------------------------------------------------
// Phase 1: each block scans its own chunk independently.
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

    __shared__ Element s_total;
    ScanImpl::block_scan(shared_data, CHUNK_SIZE, &s_total);
    __syncthreads();

    if (global_idx < L)
        output[global_idx] = shared_data[tid];

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
// chunked_scan_impl — internal recursive implementation using a pre-allocated
// scratch buffer. The scratch pointer is advanced at each recursion level so
// no cudaMalloc/cudaFree ever happens inside the timed region.
//
// Scratch layout (all levels combined, sizes in Elements):
//   Level 1: num_blocks      = L / CHUNK_SIZE
//   Level 2: num_blocks²     = L / CHUNK_SIZE²
//   ...
//   Total:   < L / (CHUNK_SIZE - 1)  << L
//
// So a scratch buffer of L Elements (allocated once by caller) covers all
// levels with room to spare even at CHUNK_SIZE=16.
// ---------------------------------------------------------------------------
template <typename ScanImpl>
void chunked_scan_impl(
    Element* d_input,
    Element* d_output,
    int L,
    Element* d_scratch)   // pre-allocated, must hold at least num_blocks Elements
{
    if (L <= 0) return;

    int num_blocks = (L + CHUNK_SIZE - 1) / CHUNK_SIZE;
    size_t shmem = shmem_elements<ScanImpl>() * sizeof(Element);

    // Carve block_totals from the front of scratch; advance scratch pointer
    // for the next recursion level.
    Element* d_block_totals  = d_scratch;
    Element* d_scratch_next  = d_scratch + num_blocks;  // next level uses remainder

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
        // Scan block_totals in-place using a temporary output buffer carved
        // from scratch. After the recursive call, d_scratch_next holds the
        // scanned totals; copy back to d_block_totals.
        Element* d_scanned_totals = d_scratch_next + num_blocks;
        chunked_scan_impl<ScanImpl>(
            d_block_totals, d_scanned_totals, num_blocks, d_scratch_next);
        cudaMemcpy(d_block_totals, d_scanned_totals,
                   num_blocks * sizeof(Element), cudaMemcpyDeviceToDevice);
    }

    // Phase 3
    phase3_propagate<<<num_blocks, CHUNK_SIZE>>>(d_output, d_block_totals, L);
}

// ---------------------------------------------------------------------------
// chunked_scan — public interface.
//
// Two variants:
//
//   chunked_scan<ScanImpl>(d_in, d_out, L)
//     Allocates scratch internally. Convenient but includes cudaMalloc cost.
//     Use for correctness testing or one-off calls.
//
//   chunked_scan<ScanImpl>(d_in, d_out, L, d_scratch)
//     Uses caller-provided scratch buffer (>= L Elements).
//     No allocation inside — suitable for benchmarking.
//     Pre-allocate once with: cudaMalloc(&d_scratch, L * sizeof(Element))
// ---------------------------------------------------------------------------
template <typename ScanImpl>
void chunked_scan(Element* d_input, Element* d_output, int L)
{
    // Convenience wrapper: allocate scratch internally
    Element* d_scratch;
    cudaMalloc(&d_scratch, L * sizeof(Element));
    chunked_scan_impl<ScanImpl>(d_input, d_output, L, d_scratch);
    cudaFree(d_scratch);
}

template <typename ScanImpl>
void chunked_scan(Element* d_input, Element* d_output, int L, Element* d_scratch)
{
    // Fast path: caller owns scratch, no allocation here
    chunked_scan_impl<ScanImpl>(d_input, d_output, L, d_scratch);
}

#endif // CHUNKED_HIERARCHICAL_RECURSIVE_CUH
