#ifndef COMMON_CUH
#define COMMON_CUH

// ---------------------------------------------------------------------------
// D is now a RUNTIME parameter — no -DD= compile flag needed.
// A single binary handles any hidden-state dimension.
//
// Why this is possible with the scalar design:
//   D is just a grid dimension (blockIdx.y).  The kernels never loop over D,
//   never store D floats per thread, and shared memory is fixed at 16 KB
//   regardless of D.  So D can be passed as a plain int at launch time.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CHUNK_SIZE: timesteps processed per block, per dimension.
// Fixed at 1024 regardless of D — shared memory cost is always 16 KB/block.
// ---------------------------------------------------------------------------
#define CHUNK_SIZE 1024

// ---------------------------------------------------------------------------
// Data layout: dimension-major, two separate float arrays.
//   a_ptr[d * L + t]   b_ptr[d * L + t]    for d in [0,D), t in [0,L)
//
// Threads in a warp all share the same d (blockIdx.y) and hold consecutive
// t values (threadIdx.x) → perfectly coalesced HBM access.
// ---------------------------------------------------------------------------

// Shared-memory layout (4 * CHUNK_SIZE floats = 16 KB, same for every D):
//   shmem[0            .. CHUNK_SIZE)   -> s_a   (primary a-values)
//   shmem[CHUNK_SIZE   .. 2*CHUNK_SIZE) -> s_b   (primary b-values)
//   shmem[2*CHUNK_SIZE .. 3*CHUNK_SIZE) -> aux_a (HillisSteele double-buffer)
//   shmem[3*CHUNK_SIZE .. 4*CHUNK_SIZE) -> aux_b (HillisSteele double-buffer)
#define SHMEM_FLOATS  (4 * CHUNK_SIZE)
#define SHMEM_BYTES   (SHMEM_FLOATS * sizeof(float))

#endif // COMMON_CUH
