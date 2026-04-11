/**
 * run_reference.cpp — Sequential CPU SSM prefix scan baseline.
 *
 * Loads inputs from SyntheticData/inputs/, runs sequential scan,
 * saves reference outputs to SequentialData/ for kernel correctness checking,
 * and prints timing results.
 *
 * Build:  g++ -O2 -std=c++17 -o run_reference run_reference.cpp
 * Usage:  ./run_reference [input_dir] [output_dir]
 *         defaults: SyntheticData/inputs   SequentialData
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <sys/stat.h>

static const int BATCH_SIZE    = 1;
static const int SEQ_LENGTHS[] = { 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072 };
static const int HIDDEN_DIMS[] = { 16, 64, 256, 512 };
static const int N_L = sizeof(SEQ_LENGTHS) / sizeof(SEQ_LENGTHS[0]);
static const int N_D = sizeof(HIDDEN_DIMS)  / sizeof(HIDDEN_DIMS[0]);
static const int N_WARMUP = 2;
static const int N_REPEAT = 5;

static bool load_inputs(const char* path, float* a, float* b, size_t n) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); return false; }
    bool ok = fread(a, sizeof(float), n, f) == n
           && fread(b, sizeof(float), n, f) == n;
    fclose(f);
    return ok;
}

static bool save_output(const char* path, const float* out, size_t n) {
    FILE* f = fopen(path, "wb");
    if (!f) { perror(path); return false; }
    fwrite(out, sizeof(float), n, f);
    fclose(f);
    return true;
}

static void sequential_scan(
    const float* a, const float* b, float* out,
    int B, int L, int D)
{
    const size_t seq_stride = (size_t)L * D;
    for (int batch = 0; batch < B; batch++) {
        const float* ab = a + batch * seq_stride;
        const float* bb = b + batch * seq_stride;
              float* xb = out + batch * seq_stride;
        for (int d = 0; d < D; d++)
            xb[d] = bb[d];
        for (int t = 1; t < L; t++) {
            for (int d = 0; d < D; d++)
                xb[t*D+d] = ab[t*D+d] * xb[(t-1)*D+d] + bb[t*D+d];
        }
    }
}

using Clock = std::chrono::high_resolution_clock;

static double median_ms(const float* a, const float* b, float* out, int B, int L, int D) {
    for (int i = 0; i < N_WARMUP; i++) sequential_scan(a, b, out, B, L, D);
    double t[N_REPEAT];
    for (int r = 0; r < N_REPEAT; r++) {
        auto t0 = Clock::now();
        sequential_scan(a, b, out, B, L, D);
        auto t1 = Clock::now();
        t[r] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    for (int i = 1; i < N_REPEAT; i++) {
        double k = t[i]; int j = i-1;
        while (j >= 0 && t[j] > k) { t[j+1] = t[j]; j--; }
        t[j+1] = k;
    }
    return t[N_REPEAT/2];
}

int main(int argc, char* argv[]) {
    const char* indir  = (argc > 1) ? argv[1] : "../SyntheticData/inputs";
    const char* outdir = (argc > 2) ? argv[2] : "SequentialData";
    mkdir(outdir, 0755);

    printf("Sequential SSM Prefix Scan Baseline\n");
    printf("B=%d  warmup=%d  repeat=%d (median)\n", BATCH_SIZE, N_WARMUP, N_REPEAT);
    printf("input:  %s\n", indir);
    printf("output: %s\n\n", outdir);
    printf("%-10s %-6s %-14s %-12s\n", "L", "D", "time_ms", "BW_GB/s");
    printf("%.46s\n", "----------------------------------------------");

    for (int li = 0; li < N_L; li++) {
        for (int di = 0; di < N_D; di++) {
            int L = SEQ_LENGTHS[li];
            int D = HIDDEN_DIMS[di];
            int B = BATCH_SIZE;
            size_t n = (size_t)B * L * D;

            std::vector<float> a(n), b(n), out(n);

            char inpath[256];
            snprintf(inpath, sizeof(inpath), "%s/input_B%d_L%d_D%d.bin", indir, B, L, D);
            if (!load_inputs(inpath, a.data(), b.data(), n)) {
                fprintf(stderr, "Missing input file — run generate_inputs first.\n");
                return 1;
            }

            double ms = median_ms(a.data(), b.data(), out.data(), B, L, D);
            double bw = (3.0 * n * sizeof(float)) / (ms * 1e-3) * 1e-9;

            // Save reference output for GPU kernel correctness checking
            char outpath[256];
            snprintf(outpath, sizeof(outpath), "%s/ref_B%d_L%d_D%d.bin", outdir, B, L, D);
            if (!save_output(outpath, out.data(), n)) {
                fprintf(stderr, "Failed to save %s\n", outpath);
                return 1;
            }

            printf("%-10d %-6d %-14.3f %-12.2f\n", L, D, ms, bw);
            fflush(stdout);
        }
    }

    printf("\nDone. Reference outputs saved to %s/\n", outdir);
    return 0;
}