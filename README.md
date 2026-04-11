# Parallel Prefix Scan Benchmarking for Mamba SSM

We are benchmarking four parallel prefix scan variants applied to the selective scan step of the Mamba State Space Model. The selective scan computes the linear recurrence:

```
x_t = Ā_t · x_{t-1} + B̄_t · u_t
```

Because this recurrence is linear, it admits an associative binary operator, enabling parallel prefix scan.

## Shared Interface (common.cuh)

### Element struct

`D` is the SSM hidden state dimension (compile-time constant). We sweep D = 16, 64, 256, 512.

```cuda
struct Element {
    float A[D][D];    // D×D state transition matrix
    float b[D];       // D-dimensional vector
};
```

### Associative combine operator

```cuda
__device__ Element combine(Element left, Element right) {
    Element result;
    // result.A = right.A * left.A           (D×D matrix multiply)
    // result.b = right.A * left.b + right.b (mat-vec multiply + vec add)
    return result;
}
```

## block_scan — The Interface You Must Implement

Each teammate must define a struct with a static `block_scan` method matching this signature:

```cuda
struct YourScanName {
    static __device__ void block_scan(
        Element* shared_data,   // in-place: input on entry, scanned output on exit
        int n,                  // number of elements in this block
        Element* block_total    // output: total of all elements combined
    );
};
```

**Contract:**
- `shared_data` is in shared memory, length `n`
- On entry: `[e0, e1, e2, ..., e_{n-1}]`
- On exit: `[e0, e0⊕e1, e0⊕e1⊕e2, ..., e0⊕...⊕e_{n-1}]` (inclusive prefix scan)
- `block_total` is set to `e0⊕e1⊕...⊕e_{n-1}` (same as the last scanned element)

### Concrete struct names

```cuda
struct HillisSteele { static __device__ void block_scan(Element*, int, Element*); };
struct Blelloch     { static __device__ void block_scan(Element*, int, Element*); };
struct WarpShuffle  { static __device__ void block_scan(Element*, int, Element*); };
```

These plug directly into the chunked hierarchical wrapper as template parameters:

```cuda
chunked_scan<HillisSteele><<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, L);
chunked_scan<Blelloch><<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, L);
chunked_scan<WarpShuffle><<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, L);
```

## File Structure

| File | Owner | Description |
|------|-------|-------------|
| `common.cuh` | Shared | Element struct, combine operator, identity element |
| `hillis_steele.cu` | Teammate 1 | Hillis-Steele scan |
| `blelloch.cu` | Teammate 2 | Blelloch scan |
| `warp_shuffle.cu` | Teammate 3 | Warp-shuffle scan |
| `chunked_hierarchical.cu` | Ernest | Chunked hierarchical wrapper |
| `benchmark.cu` | Shared | Timing harness with CUDA events |
| `validation.cu` | Shared | CPU reference + correctness checks |

## Benchmarking Parameters

- **Hidden state dimensions (D):** 16, 64, 256, 512
- **Sequence lengths (L):** 1024, 4096, and larger
- **Profiling tool:** Nsight Compute (`ncu --set full ./binary`)
