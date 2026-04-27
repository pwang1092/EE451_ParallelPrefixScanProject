/**
 * hillis_steele.cu  —  Hillis-Steele inclusive prefix scan (scalar edition)
 *
 * Each thread handles ONE float pair (a_val, b_val) for one (timestep, dim)
 * coordinate.  No D-loop anywhere.
 *
 * Algorithm: ceil(log2 n) steps.  In each step s:
 *   out[i] = (i >= s) ? combine(in[i-s], in[i]) : in[i]
 * Requires double-buffering because reading in[i-s] and writing out[i] may
 * overlap across threads within the same step.
 *
 * Shared memory layout (passed by phase1 / phase2):
 *   shmem[0              .. CHUNK_SIZE) = s_a   (active buffer for a, starts here)
 *   shmem[CHUNK_SIZE     .. 2*CHUNK_SIZE) = s_b  (active buffer for b)
 *   shmem[2*CHUNK_SIZE   .. 3*CHUNK_SIZE) = aux_a (ping-pong buffer)
 *   shmem[3*CHUNK_SIZE   .. 4*CHUNK_SIZE) = aux_b
 *
 * After block_scan returns, results are always in s_a / s_b (shmem[0..n) and
 * shmem[CHUNK_SIZE..CHUNK_SIZE+n)).
 */

#include "common.cuh"

struct HillisSteele {
    // block_scan — called from phase1 and phase2 kernels.
    //   shmem   : 4*CHUNK_SIZE floats (see layout above); pre-loaded by caller.
    //   n       : number of valid elements (pad with identity beyond n).
    //   s_tot_a / s_tot_b : __shared__ pointers; set to the combined total of
    //             all n elements; readable by all threads after return.
    static __device__ void block_scan(
        float* __restrict__ shmem, int n,
        float* s_tot_a, float* s_tot_b)
    {
        // Current and next (ping-pong) buffer pointers — stored in registers,
        // same value in every thread so swapping is deterministic.
        float* a_cur = shmem;
        float* a_nxt = shmem + 2 * CHUNK_SIZE;   // aux_a
        float* b_cur = shmem + CHUNK_SIZE;
        float* b_nxt = shmem + 3 * CHUNK_SIZE;   // aux_b

        const int tid = threadIdx.x;

        // ceil(log2 n) steps
        for (int offset = 1; offset < n; offset <<= 1) {
            __syncthreads();
            if (tid < n) {
                if (tid >= offset) {
                    float la = a_cur[tid - offset], lb = b_cur[tid - offset];
                    float ra = a_cur[tid],           rb = b_cur[tid];
                    // combine(left=(la,lb), right=(ra,rb))
                    a_nxt[tid] = ra * la;
                    b_nxt[tid] = ra * lb + rb;
                } else {
                    a_nxt[tid] = a_cur[tid];
                    b_nxt[tid] = b_cur[tid];
                }
            }
            // Swap pointers (all threads do this identically)
            float* tmp;
            tmp = a_cur; a_cur = a_nxt; a_nxt = tmp;
            tmp = b_cur; b_cur = b_nxt; b_nxt = tmp;
        }
        __syncthreads();

        // If an odd number of steps ran, result landed in aux_a/aux_b.
        // Copy back to s_a/s_b so the caller always reads from a fixed location.
        if (a_cur != shmem && tid < n) {
            shmem[tid]              = a_cur[tid];
            shmem[CHUNK_SIZE + tid] = b_cur[tid];
        }
        __syncthreads();

        // Broadcast block total (last valid element) to caller via __shared__ ptr.
        if (tid == n - 1) {
            *s_tot_a = shmem[tid];
            *s_tot_b = shmem[CHUNK_SIZE + tid];
        }
        __syncthreads();
    }
};
