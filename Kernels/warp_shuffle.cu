/**
 * warp_shuffle.cu  —  Warp-shuffle inclusive prefix scan (scalar edition)
 *
 * Each thread holds exactly 2 floats (va, vb) in registers — no D-loop,
 * no bank conflicts, no register spilling.
 *
 * OLD cost at D=512:  shfl_up_element shuffled 2*512=1024 floats per step
 *                     × 5 steps = 5120 __shfl_up_sync calls per warp scan.
 * NEW cost:           2 __shfl_up_sync calls per step × 5 steps = 10 total.
 *
 * Algorithm (three phases):
 *  1. Each warp runs an inclusive scan on its own 32 elements (5 shuffle steps).
 *  2. The first warp scans the 32 per-warp totals (another 5 shuffle steps).
 *  3. Each thread adds its warp's preceding prefix to its own value.
 *
 * Shared memory:
 *   s_a = shmem[0 .. CHUNK_SIZE)       read once on entry, written on exit
 *   s_b = shmem[CHUNK_SIZE .. 2*CS)
 *   wt_a[CHUNK_SIZE/32], wt_b[CHUNK_SIZE/32] — small static arrays for warp totals
 *
 * CHUNK_SIZE=1024 → 32 warps, both static arrays fit in 256 bytes total.
 */

#include "common.cuh"

struct WarpShuffle {
    static __device__ void block_scan(
        float* __restrict__ shmem, int n,
        float* s_tot_a, float* s_tot_b)
    {
        float* s_a = shmem;
        float* s_b = shmem + CHUNK_SIZE;

        // Small static arrays for inter-warp totals.
        // CHUNK_SIZE/32 = 32 slots — 256 bytes total, negligible.
        __shared__ float wt_a[CHUNK_SIZE / 32];
        __shared__ float wt_b[CHUNK_SIZE / 32];

        const int tid  = threadIdx.x;
        const int lane = tid & 31;
        const int wid  = tid >> 5;
        const int nw   = (n + 31) >> 5;   // number of warps that cover n elements

        // ----------------------------------------------------------------
        // Step 1: load into registers (identity for tid >= n)
        // ----------------------------------------------------------------
        float va = (tid < n) ? s_a[tid] : 1.0f;
        float vb = (tid < n) ? s_b[tid] : 0.0f;

        // ----------------------------------------------------------------
        // Step 2: warp-level inclusive scan — 5 steps, all in registers
        // ----------------------------------------------------------------
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            float la = __shfl_up_sync(0xffffffffu, va, off);
            float lb = __shfl_up_sync(0xffffffffu, vb, off);
            if (lane >= off) {
                // combine(left=(la,lb), right=(va,vb))
                vb = va * lb + vb;
                va = va * la;
            }
        }

        // ----------------------------------------------------------------
        // Step 3: last active lane in each warp saves its total
        // ----------------------------------------------------------------
        bool last_in_warp = (wid < nw) && ((lane == 31) || (tid == n - 1));
        if (last_in_warp) {
            wt_a[wid] = va;
            wt_b[wid] = vb;
        }
        __syncthreads();

        // ----------------------------------------------------------------
        // Step 4: first warp scans the warp totals
        // nw <= 32, so the first warp can handle all totals in one pass.
        // ----------------------------------------------------------------
        if (wid == 0) {
            float wa = (tid < nw) ? wt_a[tid] : 1.0f;
            float wb = (tid < nw) ? wt_b[tid] : 0.0f;
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                float la = __shfl_up_sync(0xffffffffu, wa, off);
                float lb = __shfl_up_sync(0xffffffffu, wb, off);
                if (lane >= off) {
                    wb = wa * lb + wb;
                    wa = wa * la;
                }
            }
            if (tid < nw) { wt_a[tid] = wa; wt_b[tid] = wb; }
        }
        __syncthreads();

        // ----------------------------------------------------------------
        // Step 5: each warp adds the inclusive prefix of all preceding warps
        // ----------------------------------------------------------------
        if (wid > 0) {
            float pa = wt_a[wid - 1], pb = wt_b[wid - 1];
            // combine(left=(pa,pb), right=(va,vb))
            float new_va = va * pa;
            float new_vb = va * pb + vb;
            va = new_va;
            vb = new_vb;
        }

        // ----------------------------------------------------------------
        // Write back to shared memory; broadcast block total
        // ----------------------------------------------------------------
        if (tid < n) { s_a[tid] = va; s_b[tid] = vb; }
        if (tid == n - 1) { *s_tot_a = va; *s_tot_b = vb; }
        __syncthreads();
    }
};
