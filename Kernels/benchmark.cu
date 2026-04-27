/**
 * benchmark.cu  —  Full (L, D) sweep benchmark, single binary.
 *
 * D is now a RUNTIME value — no -DD= compile flag, one binary for all dims.
 *
 * Build (one binary):
 *   nvcc -O3 -std=c++17 -arch=sm_80 -o benchmark benchmark.cu
 *
 * Usage:
 *   ./benchmark                        # sweeps all D in HIDDEN_DIMS[]
 *   ./benchmark --D 128                # only D=128 (any value you want)
 *   ./benchmark --D 128 --L 65536      # specific (D, L) pair
 *   ./benchmark [input_dir] [ref_dir] [output_dir]   # path overrides
 *
 * Input/reference files use the original time-major binary layout from
 * generate_inputs.cpp / run_reference.cpp (unchanged).
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <sys/stat.h>

#include "chunked_hierarchical_recursive.cuh"   // pulls in common.cuh + kernels

// ---------------------------------------------------------------------------
// Default sweep config
// ---------------------------------------------------------------------------
static const int DEFAULT_DIMS[]    = { 16, 64, 128, 256, 512, 1024 };
static const int N_DEFAULT_DIMS    = (int)(sizeof(DEFAULT_DIMS) / sizeof(DEFAULT_DIMS[0]));
static const int SEQ_LENGTHS[]     = { 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072 };
static const int N_L               = (int)(sizeof(SEQ_LENGTHS) / sizeof(SEQ_LENGTHS[0]));
static const int B_BATCH           = 1;
static const int N_WARMUP          = 3;
static const int N_REPEAT          = 10;

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------
static bool load_inputs(const char* path, float* a, float* b, size_t n) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); return false; }
    bool ok = fread(a, sizeof(float), n, f) == n
           && fread(b, sizeof(float), n, f) == n;
    fclose(f); return ok;
}
static bool load_ref(const char* path, float* x, size_t n) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); return false; }
    bool ok = fread(x, sizeof(float), n, f) == n;
    fclose(f); return ok;
}

// ---------------------------------------------------------------------------
// Layout helpers (CPU, outside timed region)
// ---------------------------------------------------------------------------
static void to_dim_major(const float* src, float* dst, int L, int D) {
    for (int t = 0; t < L; t++)
        for (int d = 0; d < D; d++)
            dst[d * L + t] = src[t * D + d];
}

// ---------------------------------------------------------------------------
// Correctness check
// ---------------------------------------------------------------------------
static bool check_output(const float* b_out_dm, const float* ref,
                          int L, int D, float tol = 1e-3f) {
    for (int t = 0; t < L; t++)
        for (int d = 0; d < D; d++) {
            float gpu = b_out_dm[d * L + t];
            float cpu = ref[t * D + d];
            float err = fabsf(gpu - cpu);
            float rel = err / fmaxf(fabsf(cpu), 1e-6f);
            if (err > tol && rel > tol) {
                printf("  MISMATCH t=%d d=%d gpu=%.6f ref=%.6f "
                       "abserr=%.2e relerr=%.2e\n", t, d, gpu, cpu, err, rel);
                return false;
            }
        }
    return true;
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------
static float median(float* arr, int n) {
    for (int i = 1; i < n; i++) {
        float k = arr[i]; int j = i - 1;
        while (j >= 0 && arr[j] > k) { arr[j+1] = arr[j]; j--; }
        arr[j+1] = k;
    }
    return arr[n / 2];
}

// ---------------------------------------------------------------------------
// Kernel launcher wrappers — uniform signature
// ---------------------------------------------------------------------------
typedef void (*KernelFn)(float*, float*, float*, float*, int, int, float*);

static void run_warp_shuffle(float* ai, float* bi, float* ao, float* bo,
                              int L, int D, float* sc) {
    chunked_scan<WarpShuffle>(ai, bi, ao, bo, L, D, sc);
}
static void run_blelloch(float* ai, float* bi, float* ao, float* bo,
                         int L, int D, float* sc) {
    chunked_scan<Blelloch>(ai, bi, ao, bo, L, D, sc);
}
static void run_hillis_steele(float* ai, float* bi, float* ao, float* bo,
                               int L, int D, float* sc) {
    chunked_scan<HillisSteele>(ai, bi, ao, bo, L, D, sc);
}

// ---------------------------------------------------------------------------
// Benchmark one kernel at one (L, D)
// ---------------------------------------------------------------------------
static float benchmark_kernel(
    KernelFn fn,
    float* d_ai, float* d_bi, float* d_ao, float* d_bo, float* d_sc,
    const float* h_a_dm, const float* h_b_dm, const float* ref,
    int L, int D, bool* correct_out)
{
    const size_t n = (size_t)D * L;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    for (int i = 0; i < N_WARMUP; i++) {
        cudaMemcpy(d_ai, h_a_dm, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bi, h_b_dm, n * sizeof(float), cudaMemcpyHostToDevice);
        fn(d_ai, d_bi, d_ao, d_bo, L, D, d_sc);
    }
    cudaDeviceSynchronize();

    std::vector<float> h_bo(n);
    cudaMemcpy(h_bo.data(), d_bo, n * sizeof(float), cudaMemcpyDeviceToHost);
    *correct_out = check_output(h_bo.data(), ref, L, D);

    float times[N_REPEAT];
    for (int r = 0; r < N_REPEAT; r++) {
        cudaMemcpy(d_ai, h_a_dm, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bi, h_b_dm, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaEventRecord(start);
        fn(d_ai, d_bi, d_ao, d_bo, L, D, d_sc);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[r], start, stop);
    }
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return median(times, N_REPEAT);
}

// ---------------------------------------------------------------------------
// Run all kernels for one (L, D) pair
// ---------------------------------------------------------------------------
static void run_one_LD(
    int L, int D,
    const char* indir, const char* refdir,
    FILE* csv)
{
    const size_t n = (size_t)D * L;

    char inpath[256], refpath[256];
    snprintf(inpath,  sizeof(inpath),  "%s/input_B%d_L%d_D%d.bin", indir,  B_BATCH, L, D);
    snprintf(refpath, sizeof(refpath), "%s/ref_B%d_L%d_D%d.bin",   refdir, B_BATCH, L, D);

    std::vector<float> a_tm(n), b_tm(n), ref(n);
    if (!load_inputs(inpath, a_tm.data(), b_tm.data(), n)) {
        fprintf(stderr, "  Missing input: %s — skipping\n", inpath); return;
    }
    if (!load_ref(refpath, ref.data(), n)) {
        fprintf(stderr, "  Missing ref: %s — skipping\n", refpath); return;
    }

    // Convert to dimension-major on CPU (outside timed region)
    std::vector<float> a_dm(n), b_dm(n);
    to_dim_major(a_tm.data(), a_dm.data(), L, D);
    to_dim_major(b_tm.data(), b_dm.data(), L, D);

    // Allocate device buffers once, reuse across kernels
    const int num_chunks = (L + CHUNK_SIZE - 1) / CHUNK_SIZE;
    float *d_ai, *d_bi, *d_ao, *d_bo, *d_sc;
    cudaMalloc(&d_ai, n * sizeof(float));
    cudaMalloc(&d_bi, n * sizeof(float));
    cudaMalloc(&d_ao, n * sizeof(float));
    cudaMalloc(&d_bo, n * sizeof(float));
    cudaMalloc(&d_sc, 2ULL * D * L * sizeof(float));

    struct { const char* name; KernelFn fn; } kernels[] = {
        { "warp_shuffle",  run_warp_shuffle  },
        { "blelloch",      run_blelloch      },
        { "hillis_steele", run_hillis_steele },
    };

    for (auto& k : kernels) {
        bool correct = false;
        float ms = benchmark_kernel(
            k.fn, d_ai, d_bi, d_ao, d_bo, d_sc,
            a_dm.data(), b_dm.data(), ref.data(),
            L, D, &correct);

        double gb_s = 3.0 * n * sizeof(float) / (ms * 1e-3) / 1e9;
        printf("  %-16s D=%-5d L=%-8d %8.3f ms  %8.2f GB/s  %s\n",
               k.name, D, L, ms, gb_s, correct ? "PASS" : "FAIL");
        if (csv)
            fprintf(csv, "%s,%d,%d,%.4f,%d,%.3f\n",
                    k.name, D, L, ms, correct ? 1 : 0, gb_s);
        fflush(stdout);
        if (csv) fflush(csv);
    }

    cudaFree(d_ai); cudaFree(d_bi);
    cudaFree(d_ao); cudaFree(d_bo);
    cudaFree(d_sc);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    const char* indir  = "../SyntheticData/inputs";
    const char* refdir = "../SequentialBaseline/SequentialData";
    const char* outdir = "../Results";

    // Which D values and L values to sweep
    std::vector<int> dims(DEFAULT_DIMS, DEFAULT_DIMS + N_DEFAULT_DIMS);
    std::vector<int> lens(SEQ_LENGTHS,  SEQ_LENGTHS  + N_L);

    // Parse args: paths first (positional), then --D and --L flags
    int pos = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--D") == 0 && i + 1 < argc) {
            dims = { atoi(argv[++i]) };
        } else if (strcmp(argv[i], "--L") == 0 && i + 1 < argc) {
            lens = { atoi(argv[++i]) };
        } else if (argv[i][0] != '-') {
            if      (pos == 0) indir  = argv[i];
            else if (pos == 1) refdir = argv[i];
            else if (pos == 2) outdir = argv[i];
            pos++;
        }
    }

    mkdir(outdir, 0755);
    char csv_path[256];
    snprintf(csv_path, sizeof(csv_path), "%s/benchmark.csv", outdir);
    FILE* csv = fopen(csv_path, "w");
    if (csv) fprintf(csv, "kernel,D,L,time_ms,correct,throughput_GB_s\n");

    printf("SSM Prefix Scan Benchmark  CHUNK_SIZE=%d\n", CHUNK_SIZE);
    printf("Dims: ");  for (int d : dims) printf("%d ", d);
    printf("\nLens: "); for (int l : lens) printf("%d ", l);
    printf("\n\n");

    for (int D : dims) {
        printf("=== D = %d ===\n", D);
        for (int L : lens)
            run_one_LD(L, D, indir, refdir, csv);
        printf("\n");
    }

    if (csv) fclose(csv);
    printf("Results saved to %s\n", csv_path);
    return 0;
}
