#ifndef WARP_SHUFFLE_CUH
#define WARP_SHUFFLE_CUH

// BLOCK_SIZE scales down with D to stay within 96KB shared memory limit.
// Each Element is 2*D*4 bytes. BLOCK_SIZE elements + warp_totals must fit.
// Shared memory usage: (BLOCK_SIZE + BLOCK_SIZE/32 + 1) * 2 * D * 4 bytes
//   D=16:  (256 + 9)  * 128  = ~33KB  ✓
//   D=64:  (128 + 5)  * 512  = ~66KB  ✓
//   D=256: (32  + 2)  * 2048 = ~68KB  ✓
//   D=512: (16  + 1)  * 4096 = ~68KB  ✓
#ifdef BLOCK_SIZE
#undef BLOCK_SIZE
#endif

#if D <= 16
#define BLOCK_SIZE 256
#elif D <= 64
#define BLOCK_SIZE 128
#elif D <= 256
#define BLOCK_SIZE 32
#else
#define BLOCK_SIZE 16
#endif

#endif // WARP_SHUFFLE_CUH
