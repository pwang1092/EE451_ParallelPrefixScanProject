/**
 * benchmark.cu — Full (L, D) sweep benchmark for all 4 SSM prefix scan kernels.
 *
 * Measures wall-clock kernel time via CUDA events across all (L, D) combinations.
 * Loads inputs from SyntheticData/inputs/ and validates against reference outputs.
 * Outputs a CSV file for plotting.
 *
 * Build (one per D value):
 *   nvcc -O2 -std=c++17 -arch=sm_80 -DD=16  --maxrregcount=64 -o benchmark_D16  benchmark.cu
 *   nvcc -O2 -std=c++17 -arch=sm_80 -DD=64  --maxrregcount=64 -o benchmark_D64  benchmark.cu
 *   nvcc -O2 -std=c++17 -arch=sm_80 -DD=256 --maxrregcount=64 -o benchmark_D256 benchmark.cu
 *   nvcc -O2 -std=c++17 -arch=sm_80 -DD=512 --maxrregcount=64 -o benchmark_D512 benchmark.cu
 *
 * Usage:
 *   ./benchmark_D16 [input_dir] [ref_dir] [output_csv]
 *   defaults: ../SyntheticData/inputs
 *             ../SequentialBaseline/SequentialData
 *             ../Results/benchmark_D16.csv
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <sys/stat.h>

// Include all kernels — CHUNK_SIZE defined in chunked_hierarchical.cuh
#include "warp_shuffle.cu"
#include "blelloch.cu"
#include "hillis_steele.cu"
#include "chunked_hierarchical_recursive.cuh"

// ---------------------------------------------------------------------------
// Sweep config — must match generate_inputs.cpp
// ---------------------------------------------------------------------------
static const int B = 1;
static const int SEQ_LENGTHS[] = { 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072 };
static const int N_L = sizeof(SEQ_LENGTHS) / sizeof(SEQ_LENGTHS[0]);
static const int N_WARMUP = 3;
static const int N_REPEAT = 10;

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------
static bool load_inputs(const char* path, float* a, float* b, size_t n) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); return false; }
    bool ok = fread(a, sizeof(float), n, f) == n
           && fread(b, sizeof(float), n, f) == n;
    fclose(f);
    return ok;
}

static bool load_ref(const char* path, float* x, size_t n) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); return false; }
    bool ok = fread(x, sizeof(float), n, f) == n;
    fclose(f);
    return ok;
}

// ---------------------------------------------------------------------------
// Correctness check — compare Element.b[d] against reference x[t,d]
// ---------------------------------------------------------------------------
static bool check_output(const Element* gpu_out, const float* ref, int L,
                         float tol = 1e-3f) {
    for (int t = 0; t < L; t++)
        for (int d = 0; d < D; d++) {
            float err = fabsf(gpu_out[t].b[d] - ref[t * D + d]);
            float rel = err / fmaxf(fabsf(ref[t * D + d]), 1e-6f);
            if (err > tol && rel > tol) return false;
        }
    return true;
}

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------
static float time_kernel_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

static float median(float* arr, int n) {
    // insertion sort
    for (int i = 1; i < n; i++) {
        float k = arr[i]; int j = i - 1;
        while (j >= 0 && arr[j] > k) { arr[j+1] = arr[j]; j--; }
        arr[j+1] = k;
    }
    return arr[n/2];
}

// ---------------------------------------------------------------------------
// Kernel launcher wrappers — uniform interface for timing
// ---------------------------------------------------------------------------

// WarpShuffle
static void run_warp_shuffle(Element* d_in, Element* d_out, int L) {
    chunked_scan<WarpShuffle>(d_in, d_out, L);
}

// Blelloch
static void run_blelloch(Element* d_in, Element* d_out, int L) {
    chunked_scan<Blelloch>(d_in, d_out, L);
}

// HillisSteele
static void run_hillis_steele(Element* d_in, Element* d_out, int L) {
    chunked_scan<HillisSteele>(d_in, d_out, L);
}

// ---------------------------------------------------------------------------
// Benchmark one kernel at one (L, D) config
// Returns median kernel time in ms, -1 if correctness fails
// ---------------------------------------------------------------------------
typedef void (*KernelFn)(Element*, Element*, int);

static float benchmark_kernel(
    KernelFn fn,
    Element* d_in, Element* d_out,
    const Element* h_in,
    const float* ref,
    int L,
    bool* correct_out)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int i = 0; i < N_WARMUP; i++) {
        cudaMemcpy(d_in, h_in, L * sizeof(Element), cudaMemcpyHostToDevice);
        fn(d_in, d_out, L);
    }
    cudaDeviceSynchronize();

    // Correctness check on last warmup output
    std::vector<Element> h_out(L);
    cudaMemcpy(h_out.data(), d_out, L * sizeof(Element), cudaMemcpyDeviceToHost);
    *correct_out = check_output(h_out.data(), ref, L);

    // Timed runs
    float times[N_REPEAT];
    for (int r = 0; r < N_REPEAT; r++) {
        cudaMemcpy(d_in, h_in, L * sizeof(Element), cudaMemcpyHostToDevice);
        cudaEventRecord(start);
        fn(d_in, d_out, L);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        times[r] = time_kernel_ms(start, stop);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return median(times, N_REPEAT);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    const char* indir   = (argc > 1) ? argv[1] : "../SyntheticData/inputs";
    const char* refdir  = (argc > 2) ? argv[2] : "../SequentialBaseline/SequentialData";
    const char* outdir  = (argc > 3) ? argv[3] : "../Results";

    mkdir(outdir, 0755);

    // Output CSV path
    char csv_path[256];
    snprintf(csv_path, sizeof(csv_path), "%s/benchmark_D%d.csv", outdir, D);
    FILE* csv = fopen(csv_path, "w");
    if (!csv) { perror(csv_path); return 1; }
    fprintf(csv, "kernel,D,L,time_ms,correct,throughput_GB_s\n");

    printf("SSM Prefix Scan Benchmark  D=%d  CHUNK_SIZE=%d\n\n", D, CHUNK_SIZE);
    printf("%-16s %-8s %-12s %-12s %-8s\n", "kernel", "L", "time_ms", "GB/s", "correct");
    printf("%.60s\n", "------------------------------------------------------------");

    struct { const char* name; KernelFn fn; } kernels[] = {
        { "warp_shuffle",   run_warp_shuffle   },
        { "blelloch",       run_blelloch       },
        { "hillis_steele",  run_hillis_steele  },
    };
    const int N_KERNELS = 3;

    for (int li = 0; li < N_L; li++) {
        int L = SEQ_LENGTHS[li];
        size_t n = (size_t)L * D;

        // Load inputs
        char inpath[256];
        snprintf(inpath, sizeof(inpath), "%s/input_B%d_L%d_D%d.bin", indir, B, L, D);
        std::vector<float> a(n), b(n);
        if (!load_inputs(inpath, a.data(), b.data(), n)) {
            fprintf(stderr, "Missing input file: %s\n", inpath);
            return 1;
        }

        // Load reference
        char refpath[256];
        snprintf(refpath, sizeof(refpath), "%s/ref_B%d_L%d_D%d.bin", refdir, B, L, D);
        std::vector<float> ref(n);
        if (!load_ref(refpath, ref.data(), n)) {
            fprintf(stderr, "Missing ref file: %s — run run_reference first\n", refpath);
            return 1;
        }

        // Pack into Element array
        std::vector<Element> h_in(L);
        for (int t = 0; t < L; t++)
            for (int d = 0; d < D; d++) {
                h_in[t].a[d] = a[t * D + d];
                h_in[t].b[d] = b[t * D + d];
            }

        // Allocate device buffers
        Element *d_in, *d_out;
        cudaMalloc(&d_in,  L * sizeof(Element));
        cudaMalloc(&d_out, L * sizeof(Element));

        for (int ki = 0; ki < N_KERNELS; ki++) {
            bool correct = false;
            float ms = benchmark_kernel(
                kernels[ki].fn, d_in, d_out,
                h_in.data(), ref.data(), L, &correct);

            // Throughput: read a,b + write x = 3 * L * D * 4 bytes
            double bytes = 3.0 * n * sizeof(float);
            double gb_s = bytes / (ms * 1e-3) * 1e-9;

            printf("%-16s %-8d %-12.3f %-12.2f %-8s\n",
                   kernels[ki].name, L, ms, gb_s,
                   correct ? "PASS" : "FAIL");

            fprintf(csv, "%s,%d,%d,%.4f,%d,%.3f\n",
                    kernels[ki].name, D, L, ms,
                    correct ? 1 : 0, gb_s);
            fflush(csv);
        }

        cudaFree(d_in);
        cudaFree(d_out);
        printf("\n");
    }

    fclose(csv);
    printf("Results saved to %s\n", csv_path);
    return 0;
}
