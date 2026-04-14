## Setup

### 1. Generate synthetic inputs (run once)
Inputs are not committed to the repo. Generate them locally:
```bash
cd SyntheticData
g++ -O2 -std=c++17 -o generate_inputs generate_inputs.cpp
./generate_inputs
```
This writes 32 binary files (`input_B1_L*_D*.bin`) sweeping all (L, D) combinations. Each file contains `[a | b]` concatenated — transition scalars `a` uniform in [0.5, 0.99] and input projections `b` ~ N(0,1). The `a` range keeps chains numerically stable at L=131072.

### 2. Run sequential baseline
```bash
cd SequentialBaseline
sbatch run_reference.sh
```
Outputs timing results and saves reference binaries to `SequentialData/ref_B1_L*_D*.bin` for GPU kernel correctness checking.

### 3. Inspect binary files
```bash
# from project root
python3 inspect_bin.py SyntheticData/inputs/input_B1_L1024_D16.bin --type input
python3 inspect_bin.py SequentialBaseline/SequentialData/ref_B1_L1024_D16.bin --type ref
```

## Correctness Checking
GPU kernels load from `SyntheticData/inputs/` and diff their output against `SequentialBaseline/SequentialData/ref_B1_L*_D*.bin`. Both use float32 with tolerance 1e-4.

## Shared Interface (common.cuh)

### Element struct

`D` is the SSM hidden state dimension (compile-time constant). We sweep D = 16, 64, 256, 512. Because Mamba uses a diagonal Ā, the state transition is an elementwise multiply rather than a full matrix-vector multiply, so `Element` stores only the diagonal as a vector.
```cuda
struct Element {
    float a[D];    // diagonal of Ā (one scalar per hidden dimension)
    float b[D];    // B̄ · u_t (input projection, D-dimensional vector)
};
```

### Associative combine operator
```cuda
__device__ Element combine(Element left, Element right) {
    Element result;
    // result.a[d] = right.a[d] * left.a[d]              (elementwise)
    // result.b[d] = right.a[d] * left.b[d] + right.b[d] (elementwise)
    return result;
}
```

## block_scan — COMMON INTERFACE BETWEEN ALL THREE FILES

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
| `hillis_steele.cu` | Aryan | Hillis-Steele scan |
| `blelloch.cu` | Benjamin | Blelloch scan |
| `warp_shuffle.cu` | Peter | Warp-shuffle scan |
| `chunked_hierarchical.cu` | Ernest | Chunked hierarchical wrapper |
| `benchmark.cu` | Shared | Timing harness with CUDA events |
| `validation.cu` | Shared | CPU reference + correctness checks |

## Benchmarking Parameters

- **Batch size (B):** 1
- **Hidden state dimensions (D):** 16, 64, 256, 512
- **Sequence lengths (L):** 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072
- **Profiling tool:** Nsight Compute (`ncu --set full ./binary`)
