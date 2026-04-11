/**
 * generate_inputs.cpp — Write synthetic SSM scan inputs to disk.
 * Build:  g++ -O2 -std=c++17 -o bin/generate_inputs generate_inputs.cpp
 * Usage:  ./bin/generate_inputs [output_dir]   (default: ./inputs)
 */

#include <cstdio>
#include <cstdlib>
#include <random>
#include <sys/stat.h>

static const int BATCH_SIZE    = 1;
static const int SEQ_LENGTHS[] = { 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072 };
static const int HIDDEN_DIMS[] = { 16, 64, 256, 512 };
static const int N_L = sizeof(SEQ_LENGTHS) / sizeof(SEQ_LENGTHS[0]);
static const int N_D = sizeof(HIDDEN_DIMS)  / sizeof(HIDDEN_DIMS[0]);

int main(int argc, char* argv[]) {
    const char* outdir = (argc > 1) ? argv[1] : "./inputs";
    mkdir(outdir, 0755);

    for (int li = 0; li < N_L; li++) {
        for (int di = 0; di < N_D; di++) {
            int L = SEQ_LENGTHS[li];
            int D = HIDDEN_DIMS[di];
            int B = BATCH_SIZE;
            size_t n = (size_t)B * L * D;

            float* a = (float*)malloc(n * sizeof(float));
            float* b = (float*)malloc(n * sizeof(float));

            // per-config seed so each file is independently reproducible
            std::mt19937 rng(0x451u ^ ((unsigned)L << 16 | D));
            std::uniform_real_distribution<float> uniform(0.5f, 0.99f);
            std::normal_distribution<float>       normal(0.f, 1.f);

            for (size_t i = 0; i < n; i++) a[i] = uniform(rng);
            for (size_t i = 0; i < n; i++) b[i] = normal(rng);

            char path[256];
            snprintf(path, sizeof(path), "%s/input_B%d_L%d_D%d.bin", outdir, B, L, D);
            FILE* f = fopen(path, "wb");
            fwrite(a, sizeof(float), n, f);
            fwrite(b, sizeof(float), n, f);
            fclose(f);

            printf("wrote %s  (%.0f MB)\n", path, 2.0 * n * sizeof(float) / (1<<20));

            free(a); free(b);
        }
    }
    return 0;
}
