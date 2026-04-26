/**
 * blelloch.cu — Blelloch up-sweep + down-sweep algorithm for prefix scan
 *
 * Implements Blelloch::block_scan per the common.cuh interface.
 *
 * Algorithm overview:
 *   1. Up-sweep (reduce): combine pairs at stride 1, 2, 4, ... until root
 *      holds total of all elements. O(N) work, log(N) steps.
 *   2. Set root to identity (converts to exclusive scan).
 *   3. Down-sweep (distribute): push partial sums back down the tree.
 *      O(N) work, log(N) steps.
 *   4. Convert exclusive -> inclusive by combining with saved originals.
 *   5. Copy results from conflict-free (phys) positions to direct positions
 *      so the generic chunked wrapper can read shared_data[tid] correctly.
 *
 * Note on phys() indexing:
 *   During computation, data is stored at shared_data[phys(i)] to avoid
 *   shared memory bank conflicts. At the end of block_scan, results are
 *   copied to shared_data[i] (direct) so callers can access them uniformly.
 */

#include "common.cuh"       // Element, combine, identity — must come first
#include "blelloch.cuh"     // undef + redefines BLOCK_SIZE to D-scaled value

// Conflict-free shared memory indexing to avoid bank conflicts
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(i) ((i >> LOG_NUM_BANKS) + (i >> (2 * LOG_NUM_BANKS)))

__device__ __forceinline__ int phys(int i) {
    return i + CONFLICT_FREE_OFFSET(i);
}

struct Blelloch {
    static __device__ void block_scan(Element* shared_data, int n, Element* block_total) {
        int tid = threadIdx.x;
        int offset = 1;

        // Save originals from the physical shared-memory locations
        int ai = tid;
        int bi = tid + (n >> 1);
        Element originalA = shared_data[phys(ai)];
        Element originalB = shared_data[phys(bi)];

        // Up-sweep (reduce)
        for (int d = n >> 1; d > 0; d >>= 1) {
            __syncthreads();
            if (tid < d) {
                int aii = phys(offset * (2 * tid + 1) - 1);
                int bii = phys(offset * (2 * tid + 2) - 1);
                shared_data[bii] = combine(shared_data[aii], shared_data[bii]);
            }
            offset <<= 1;
        }
        __syncthreads();

        if (tid == 0) {
            *block_total = shared_data[phys(n - 1)];
            shared_data[phys(n - 1)] = identity();
        }
        __syncthreads();

        // Down-sweep (distribute)
        for (int d = 1; d < n; d <<= 1) {
            offset >>= 1;
            __syncthreads();
            if (tid < d) {
                int aii = phys(offset * (2 * tid + 1) - 1);
                int bii = phys(offset * (2 * tid + 2) - 1);
                Element t = shared_data[aii];
                shared_data[aii] = shared_data[bii];
                shared_data[bii] = combine(shared_data[bii], t);
            }
        }
        __syncthreads();

        // Convert exclusive -> inclusive by combining with saved originals
        shared_data[phys(ai)] = combine(shared_data[phys(ai)], originalA);
        shared_data[phys(bi)] = combine(shared_data[phys(bi)], originalB);
        __syncthreads();

        // Copy from conflict-free positions to direct positions so callers
        // can uniformly read shared_data[tid] (e.g. in chunked phase1).
        Element tmp_a = shared_data[phys(ai)];
        Element tmp_b = shared_data[phys(bi)];
        __syncthreads();
        shared_data[ai] = tmp_a;
        shared_data[bi] = tmp_b;
        __syncthreads();
    }
};
