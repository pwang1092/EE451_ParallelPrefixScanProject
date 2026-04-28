## Setup

### 1. Generate synthetic inputs (run once)
Inputs are not committed to the repo. Generate them locally:
```bash
cd SyntheticData
g++ -O2 -std=c++17 -o generate_inputs generate_inputs.cpp
./generate_inputs
```
This writes 40 binary files (`input_B1_L*_D*.bin`) for:
- `D = {1, 16, 64, 256, 512}`
- `L = {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072}`

Each file stores `[a | b]` arrays in time-major layout (`t * D + d`).

### 2. Run sequential baseline and generate references
```bash
cd SequentialBaseline
sbatch run_reference.sh
```
This writes:
- CPU timing summary rows
- reference outputs `SequentialData/ref_B1_L*_D*.bin`

### 3. Run GPU benchmark sweep (single binary)
```bash
cd Kernels
nvcc -O3 -std=c++17 -arch=sm_80 -o benchmark benchmark.cu
./benchmark ../SyntheticData/inputs ../SequentialBaseline/SequentialData ../Results
```
Output CSV: `Results/benchmark.csv`

### 4. Inspect binary files
```bash
# from project root
python3 inspect_bin.py SyntheticData/inputs/input_B1_L1024_D16.bin --type input
python3 inspect_bin.py SequentialBaseline/SequentialData/ref_B1_L1024_D16.bin --type ref
```

## Runtime-D Scalar Kernel Design

The current implementation uses runtime `D` and scalar `(a, b)` values per thread.

- `D` is passed at runtime and mapped to `blockIdx.y`
- each thread handles one `(t, d)` coordinate
- data layout inside kernels is dimension-major: `ptr[d * L + t]`
- `CHUNK_SIZE` is fixed at `1024` for all `D`
- shared memory footprint is fixed (`SHMEM_BYTES = 16 KB`)

This avoids shrinking `CHUNK_SIZE` at large `D` and preserves sequence-length parallelism while also exposing hidden-dimension parallelism.

## block_scan Interface (per-kernel)

All three scan variants expose the same scalar shared-memory contract:
```cuda
struct WarpShuffle {
    static __device__ void block_scan(float* shmem, int n, float* s_tot_a, float* s_tot_b);
};
```
Same signature is used by `Blelloch` and `HillisSteele`.

## Benchmark and Profiling Policy (Submission)

Official sweep policy for this repo:
- `D = {1, 16, 64, 256, 512}`
- `L = {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072}`

`benchmark.cu` defaults are aligned to that policy.

## Profiling Workflow (Nsight Compute)

### Run full profiling sweep on CARC
```bash
cd Kernels
sbatch profile_all_metrics.sh
```

`profile_all_metrics.sh` is end-to-end: it first regenerates synthetic inputs,
then runs the sequential baseline to produce references, then runs Nsight
profiling for all `(kernel, D, L)` cases.

The script compiles one runtime-D profile driver and runs all `(kernel, D, L)` combinations, collecting:
- timing CSV (`timing.csv`)
- raw Nsight CSV logs (`raw_ncu/*.csv`)
- analyzed reports

Output root:
- `Results/profile_run_<jobid>/`

### Analyze outputs
The profiling script runs:
```bash
python3 Kernels/analyze_metrics.py --timing_csv ... --raw_dir ... --out_dir ...
```

Generated analysis files include:
- `merged_metrics.csv`
- `kernel_launch_metrics.csv`
- `phase_breakdown.csv`
- `occupancy_summary.csv`
- `crossover_summary.csv`
- `analysis_report.txt`

## Correctness Checking

GPU output is compared against sequential references in float32 with tolerance `1e-3`.

## Key Files

| File | Purpose |
|------|---------|
| `Kernels/common.cuh` | Runtime-D constants and shared-memory layout |
| `Kernels/warp_shuffle.cu` | Warp-shuffle scalar scan |
| `Kernels/blelloch.cu` | Blelloch scalar scan |
| `Kernels/hillis_steele.cu` | Hillis-Steele scalar scan |
| `Kernels/chunked_hierarchical_recursive.cuh` | Runtime-D hierarchical scan wrapper |
| `Kernels/benchmark.cu` | Benchmark harness (single binary, runtime-D) |
| `Kernels/profile_driver.cu` | Single-case runtime-D timing/profiling driver |
| `Kernels/profile_all_metrics.sh` | CARC profiling sweep launcher |
| `Kernels/analyze_metrics.py` | Aggregates timing + Nsight metrics |
