#ifndef HILLIS_STEELE_CUH
#define HILLIS_STEELE_CUH

// BLOCK_SIZE scales down with D to stay within 96KB shared memory limit.
// Hillis-Steele needs two full-size buffers for double-buffering (in + out).
// Total shared memory = 2 * BLOCK_SIZE * sizeof(Element) = 2 * BLOCK_SIZE * 2*D*4
// Budget = 96KB = 98304 bytes  =>  BLOCK_SIZE <= 98304 / (16 * D)
//   D=16:  <= 384  -> use 256
//   D=64:  <= 96   -> use 64
//   D=256: <= 24   -> use 16
//   D=512: <= 12   -> use 8
#ifdef BLOCK_SIZE
#undef BLOCK_SIZE
#endif

#if D <= 16
#define BLOCK_SIZE 256
#elif D <= 64
#define BLOCK_SIZE 64
#elif D <= 256
#define BLOCK_SIZE 16
#else
#define BLOCK_SIZE 8
#endif

#endif // HILLIS_STEELE_CUH
