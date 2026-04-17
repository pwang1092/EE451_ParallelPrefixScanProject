/**
 * test_blelloch.cu — Correctness test for Blelloch::block_scan
 *
 * Tests single-block scan (L <= BLOCK_SIZE) against CPU reference .bin files.
 * For L > BLOCK_SIZE you need the chunked wrapper — test those configs later.
 *
 * Build:
 *   nvcc -O2 -std=c++17 -DD=16  -o test_blelloch_D16  test_blelloch.cu
 *   nvcc -O2 -std=c++17 -DD=64  -o test_blelloch_D64  test_blelloch.cu
 *   nvcc -O2 -std=c++17 -DD=256 -o test_blelloch_D256 test_blelloch.cu
 *   nvcc -O2 -std=c++17 -DD=512 -o test_blelloch_D512 test_blelloch.cu
 *
 * Usage:
 *   ./test_blelloch_D16 [input_dir] [ref_dir]
 *   defaults: ../SyntheticData/inputs   ../SequentialBaseline/SequentialData
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "common.cuh"
#include "blelloch.cu"

// Kernel: one block, one scan
__global__ void blelloch_kernel(Element* d_in, Element* d_out, int L) {
    extern __shared__ Element shared_data[];
    __shared__ Element block_total;

    int tid = threadIdx.x;
    int half = L / 2;

    if (tid < half) {
        int ai = tid;
        int bi = tid + half;
        shared_data[phys(ai)] = d_in[ai];
        shared_data[phys(bi)] = d_in[bi];
    }
    __syncthreads();

    Blelloch::block_scan(shared_data, L, &block_total);

    if (tid < half) {
        int ai = tid;
        int bi = tid + half;
        d_out[ai] = shared_data[phys(ai)];
        d_out[bi] = shared_data[phys(bi)];
    }
}


// File I/O
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
// Compare GPU output (Element array) against reference x array
// The reference stores x[t,d] = final hidden state.
// Our Element.b[d] after the scan holds exactly x[t,d].
// ---------------------------------------------------------------------------
static bool check(const Element* gpu_out, const float* ref, int L, float tol = 1e-4f) {
    int n_errors = 0;
    float max_err = 0.f;
    for (int t = 0; t < L; t++) {
        for (int d = 0; d < D; d++) {
            float got = gpu_out[t].b[d];
            float expected = ref[t * D + d];
            float err = fabsf(got - expected);
            if (err > max_err) max_err = err;
            if (err > tol) {
                if (n_errors < 5)
                    printf("  MISMATCH t=%d d=%d  got=%.6f  expected=%.6f  err=%.2e\n",
                           t, d, got, expected, err);
                n_errors++;
            }
        }
    }
    printf("  max_err=%.2e  n_errors=%d/%d  %s\n",
           max_err, n_errors, L * D, n_errors == 0 ? "PASS" : "FAIL");
    return n_errors == 0;
}


// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    const char* indir  = (argc > 1) ? argv[1] : "SyntheticData/inputs";
    const char* refdir = (argc > 2) ? argv[2] : "SequentialBaseline/SequentialData";

    // Only test L <= BLOCK_SIZE since we have no chunked wrapper yet
    const int TEST_LENGTHS[] = { 32, 64, 128, 256 };
    const int N_L = sizeof(TEST_LENGTHS) / sizeof(TEST_LENGTHS[0]);

    printf("Blelloch correctness test  D=%d  BLOCK_SIZE=%d\n\n", D, BLOCK_SIZE);

    bool all_pass = true;

    for (int li = 0; li < N_L; li++) {
        int L = TEST_LENGTHS[li];
        int B = 1;
        size_t n = (size_t)L * D;

        // --- load inputs ---
        // Find the smallest L in our files that is >= test L
        // We generated files for L=1024 and up, so load L=1024 and use first L elements
        char inpath[256];
        snprintf(inpath, sizeof(inpath), "%s/input_B%d_L1024_D%d.bin", indir, B, D);

        std::vector<float> a(1024 * D), b(1024 * D);
        if (!load_inputs(inpath, a.data(), b.data(), 1024 * D)) {
            fprintf(stderr, "Could not load %s\n", inpath);
            return 1;
        }

        // --- load reference (also from L=1024 file, use first L elements) ---
        char refpath[256];
        snprintf(refpath, sizeof(refpath), "%s/ref_B%d_L1024_D%d.bin", refdir, B, D);
        std::vector<float> ref(1024 * D);
        if (!load_ref(refpath, ref.data(), 1024 * D)) {
            fprintf(stderr, "Could not load %s\n  Run run_reference first.\n", refpath);
            return 1;
        }

        // --- pack into Element array ---
        std::vector<Element> h_in(L), h_out(L);
        for (int t = 0; t < L; t++) {
            for (int d = 0; d < D; d++) {
                h_in[t].a[d] = a[t * D + d];
                h_in[t].b[d] = b[t * D + d];
            }
        }

        // --- copy to device ---
        Element *d_in, *d_out;
        cudaMalloc(&d_in,  L * sizeof(Element));
        cudaMalloc(&d_out, L * sizeof(Element));
        cudaMemcpy(d_in, h_in.data(), L * sizeof(Element), cudaMemcpyHostToDevice);

        // --- run kernel ---
        int shared_elems = L + CONFLICT_FREE_OFFSET(L - 1);
        size_t shared_bytes = shared_elems * sizeof(Element);
        int num_threads = L/2;
        blelloch_kernel<<<1, num_threads, shared_bytes>>>(d_in, d_out, L);
        cudaDeviceSynchronize();

        // check for kernel errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("L=%d: CUDA error: %s\n", L, cudaGetErrorString(err));
            all_pass = false;
            cudaFree(d_in); cudaFree(d_out);
            continue;
        }

        // --- copy back ---
        cudaMemcpy(h_out.data(), d_out, L * sizeof(Element), cudaMemcpyDeviceToHost);

        // --- check ---
        printf("L=%-4d D=%-4d  ", L, D);
        bool pass = check(h_out.data(), ref.data(), L);
        all_pass = all_pass && pass;

        cudaFree(d_in);
        cudaFree(d_out);
    }

    printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
