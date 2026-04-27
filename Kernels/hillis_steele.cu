/**
 * hillis_steele.cu — Hillis-Steele step-optimal parallel prefix scan
 *
 * Implements HillisSteele::block_scan per the common.cuh interface.
 *
 * Algorithm overview:
 *   Hillis-Steele is a step-optimal (ceil(log2 n) steps) but work-inefficient
 *   (O(n log n) combine operations) inclusive scan.
 *
 *   For step s = 1, 2, 4, ...,:
 *     Each thread i computes:
 *       out[i] = (i >= s) ? combine(in[i - s], in[i]) : in[i]
 *     Then the in and out buffers are swapped for the next step.
 *
 *   Double-buffering in shared memory is required because thread i reads
 *   position i-s which is owned by a different thread. Without a separate
 *   output buffer, a thread writing its result could corrupt another thread's
 *   unread input within the same step.
 *
 * Shared memory layout (dynamic, allocated by caller):
 *   The caller allocates 2 * CHUNK_SIZE * sizeof(Element) and passes a pointer
 *   to the first half as shared_data. block_scan uses shared_data[CHUNK_SIZE..]
 *   as buf_B (second buffer) via pointer arithmetic — no static declaration.
 *   This avoids the ptxas compile-time shared memory limit check.
 */

#include "common.cuh"           // Element, combine, identity — must come first
#include "hillis_steele.cuh"    // undef + redefines BLOCK_SIZE to D-scaled value

struct HillisSteele {
    static __device__ void block_scan(Element* shared_data, int n, Element* block_total) {
        // buf_B lives in the second half of the caller's dynamic shared allocation.
        // Caller must allocate 2 * BLOCK_SIZE * sizeof(Element) and pass a pointer
        // to the first half. This avoids a static __shared__ declaration which would
        // fail ptxas compile-time checks when BLOCK_SIZE * sizeof(Element) > 48KB.
        Element* buf_B = shared_data + BLOCK_SIZE;

        int tid = threadIdx.x;

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
            Element* tmp = in_buf;
            in_buf  = out_buf;
            out_buf = tmp;
        }
        __syncthreads();

        // After ceil(log2 n) steps, final results are in in_buf.
        // If steps was odd, in_buf == buf_B; copy back to shared_data
        // so the caller always reads from shared_data.
        if (in_buf != shared_data && tid < n)
            shared_data[tid] = in_buf[tid];
        __syncthreads();

        if (tid == n - 1)
            *block_total = shared_data[tid];
        __syncthreads();
    }
};
