/**
 * hillis_steele.cu — Hillis-Steele step-optimal parallel prefix scan
 *
 * Implements HillisSteele::block_scan per the common.cuh interface.
 *
 * Algorithm overview:
 *   Hillis-Steele is a step-optimal (ceil(log2 n) steps) but work-inefficient
 *   (O(n log n) combine operations) inclusive scan.
 *
 *   For step s = 1, 2, 4, ...:
 *     Each thread i computes:
 *       out[i] = (i >= s) ? combine(in[i - s], in[i]) : in[i]
 *     Then the in and out buffers are swapped for the next step.
 *
 *   Double-buffering in shared memory is required because thread i reads
 *   position i-s which is owned by a different thread. Without a separate
 *   output buffer, a thread writing its result could corrupt another thread's
 *   unread input within the same step.
 *
 * Shared memory usage:
 *   shared_data (caller-allocated) + buf_B (declared here) = 2 * BLOCK_SIZE * sizeof(Element)
 *   BLOCK_SIZE is scaled in hillis_steele.cuh so this stays within 96KB.
 */

#include "common.cuh"           // Element, combine, identity — must come first
#include "hillis_steele.cuh"    // undef + redefines BLOCK_SIZE to D-scaled value

struct HillisSteele {
    static __device__ void block_scan(Element* shared_data, int n, Element* block_total) {
        // Second buffer for double-buffering. Sized to BLOCK_SIZE (compile-time
        // constant from hillis_steele.cuh) which is always >= n.
        __shared__ Element buf_B[BLOCK_SIZE];

        int tid = threadIdx.x;

        // in_buf/out_buf are swapped each step; track which buffer holds
        // the current results so we can copy back to shared_data if needed.
        Element* in_buf  = shared_data;
        Element* out_buf = buf_B;

        for (int offset = 1; offset < n; offset <<= 1) {
            __syncthreads();
            if (tid < n) {
                if (tid >= offset)
                    out_buf[tid] = combine(in_buf[tid - offset], in_buf[tid]);
                else
                    out_buf[tid] = in_buf[tid];
            }
            // Swap for next iteration
            Element* tmp = in_buf;
            in_buf  = out_buf;
            out_buf = tmp;
        }
        __syncthreads();

        // After ceil(log2 n) steps, final results are in in_buf.
        // If the number of steps was odd, in_buf == buf_B; copy back to shared_data
        // so the caller always reads from shared_data.
        if (in_buf != shared_data && tid < n) {
            shared_data[tid] = in_buf[tid];
        }
        __syncthreads();

        // block_total = combined result of all n elements (last element of inclusive scan)
        if (tid == n - 1)
            *block_total = shared_data[tid];

        __syncthreads();
    }
};
