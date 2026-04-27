/**
 * blelloch.cu — Blelloch up-sweep + down-sweep algorithm for prefix scan
 *
 * Implements Blelloch::block_scan per the common.cuh interface.
 *
 * Thread count note:
 *   Blelloch uses n/2 threads to scan n elements (each thread handles ai and bi).
 *   phase1_local_scan launches CHUNK_SIZE threads but Blelloch only needs CHUNK_SIZE/2.
 *   Threads with bi >= n are "inactive" (have ai in valid range but bi out of range).
 *   All bi accesses are guarded with `active = (bi < n)` to prevent OOB reads/writes.
 *
 * Conflict-free addressing note:
 *   Data is stored at shared_data[phys(i)] internally to avoid bank conflicts.
 *   phase1 loads data to direct positions shared_data[tid], so we copy direct→phys
 *   at entry and phys→direct at exit to maintain a consistent external interface.
 */

#include "common.cuh"       // Element, combine, identity — must come first
#include "blelloch.cuh"     // undef + redefines BLOCK_SIZE to D-scaled value

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

        int ai = tid;
        int bi = tid + (n >> 1);
        bool active = (bi < n);  // threads 0..n/2-1 are active; rest have bi out of range

        // Copy from direct positions (loaded by caller) to conflict-free (phys) positions.
        // Inactive threads (bi >= n) only write ai position — bi is guarded.
        Element val_a = shared_data[ai];
        Element val_b = active ? shared_data[bi] : identity();
        __syncthreads();
        shared_data[phys(ai)] = val_a;
        if (active) shared_data[phys(bi)] = val_b;
        __syncthreads();

        // Save originals for inclusive conversion later
        Element originalA = shared_data[phys(ai)];
        Element originalB = active ? shared_data[phys(bi)] : identity();

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

        // Convert exclusive -> inclusive
        if (active) {
            shared_data[phys(ai)] = combine(shared_data[phys(ai)], originalA);
            shared_data[phys(bi)] = combine(shared_data[phys(bi)], originalB);
        }
        __syncthreads();

        // Copy from phys positions back to direct positions for generic access.
        // Guard bi to avoid corrupting valid data or s_total.
        Element tmp_a = shared_data[phys(ai)];
        Element tmp_b = active ? shared_data[phys(bi)] : identity();
        __syncthreads();
        shared_data[ai] = tmp_a;
        if (active) shared_data[bi] = tmp_b;
        __syncthreads();
    }
};
