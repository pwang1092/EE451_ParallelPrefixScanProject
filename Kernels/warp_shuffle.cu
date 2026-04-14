/**
 * warp_shuffle.cu — Warp-shuffle inclusive prefix scan for SSM recurrence.
 *
 * Implements WarpShuffle::block_scan per the common.cuh interface.
 *
 * Algorithm overview:
 *   1. Each thread holds one Element (one timestep's (a, b) pair).
 *   2. Within each warp (32 threads), run an inclusive scan using __shfl_up_sync
 *      in log2(32) = 5 steps — no shared memory, no bank conflicts.
 *   3. The last active thread in each warp writes its partial total to shared memory.
 *   4. Thread 0..num_warps-1 (first warp) scans the warp totals.
 *   5. Each thread adds the preceding warp's total to its own value.
 *
 * Why warp shuffle?
 *   __shfl_up_sync exchanges registers directly between threads in the same warp.
 *   No shared memory round-trip means no bank conflicts and lower latency than
 *   Blelloch or Hillis-Steele for small-to-medium D where register pressure is ok.
 *   At large D (512), each Element is 2*D*4 = 4KB of registers, which limits
 *   occupancy — this is one of the hypotheses we test in the benchmark.
 */

#include "common.cuh"

// ---------------------------------------------------------------------------
// Shuffle a full Element up by `delta` lanes within the warp.
// We must call __shfl_up_sync once per float since it only moves 32 bits.
// ---------------------------------------------------------------------------
__device__ __forceinline__ Element shfl_up_element(Element val, int delta, unsigned mask = 0xffffffffu) {
    Element result;
    #pragma unroll
    for (int d = 0; d < D; d++) {
        result.a[d] = __shfl_up_sync(mask, val.a[d], delta);
        result.b[d] = __shfl_up_sync(mask, val.b[d], delta);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Inclusive warp scan. Each thread ends up with the prefix from lane 0 to
// its own lane. Threads with lane < offset keep their own value unchanged.
// ---------------------------------------------------------------------------
__device__ __forceinline__ Element warp_inclusive_scan(Element val) {
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        Element left = shfl_up_element(val, offset);
        if (lane >= offset)
            val = combine(left, val);
    }
    return val;
}

// ---------------------------------------------------------------------------
// WarpShuffle::block_scan
//   shared_data : in shared memory, length n, modified in-place
//   n           : number of active elements (<= blockDim.x)
//   block_total : set to the combined value of all n elements
//
// Assumes one thread per element: tid = threadIdx.x handles shared_data[tid].
// blockDim.x must be a multiple of 32 (pad with identity if n < blockDim.x).
// ---------------------------------------------------------------------------
struct WarpShuffle {
    static __device__ void block_scan(
        Element* shared_data,
        int n,
        Element* block_total)
    {
        const int tid     = threadIdx.x;
        const int lane    = tid & 31;
        const int warp_id = tid >> 5;
        const int num_warps = (blockDim.x + 31) >> 5;

        // Shared storage for per-warp totals (max 32 warps = 1024 threads/block)
        __shared__ Element warp_totals[32];

        // Step 1: load element (identity for out-of-range threads)
        Element val = (tid < n) ? shared_data[tid] : identity();

        // Step 2: warp-level inclusive scan via shuffle ---
        val = warp_inclusive_scan(val);

        // Step 3: last active lane in each warp saves its total
        // "Last active" = min(lane 31, last thread in block that is < n)
        bool is_last_in_warp = (lane == 31) || (tid == n - 1);
        if (is_last_in_warp)
            warp_totals[warp_id] = val;
        __syncthreads();

        // Step 4: first warp scans the warp totals
        if (warp_id == 0) {
            Element wt = (tid < num_warps) ? warp_totals[tid] : identity();
            wt = warp_inclusive_scan(wt);
            if (tid < num_warps)
                warp_totals[tid] = wt;
        }
        __syncthreads();

        // Step 5: add preceding warp's total to every thread
        if (warp_id > 0)
            val = combine(warp_totals[warp_id - 1], val);

        // Write back
        if (tid < n)
            shared_data[tid] = val;

        // block_total = combined value of all n elements (last active element)
        if (tid == n - 1)
            *block_total = val;

        __syncthreads();
    }
};
