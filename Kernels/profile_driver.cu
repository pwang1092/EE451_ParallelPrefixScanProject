#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "chunked_hierarchical_recursive.cuh"

#define CUDA_CHECK(stmt)                                                      \
    do {                                                                      \
        cudaError_t _err = (stmt);                                            \
        if (_err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(_err));            \
            return false;                                                     \
        }                                                                     \
    } while (0)

namespace {

constexpr int kBatchSize = 1;

struct Options {
    std::string kernel;
    int D = -1;
    int L = -1;
    std::string input_dir = "../SyntheticData/inputs";
    std::string ref_dir = "../SequentialBaseline/SequentialData";
    std::string csv_append;
    int warmup = 3;
    int repeat = 10;
    bool no_print = false;
    bool skip_check = false;
};

using KernelFn = void (*)(float*, float*, float*, float*, int, int, float*);

void launch_warp_shuffle(float* d_ai, float* d_bi, float* d_ao, float* d_bo,
                         int L, int D, float* d_sc) {
    chunked_scan<WarpShuffle>(d_ai, d_bi, d_ao, d_bo, L, D, d_sc);
}

void launch_blelloch(float* d_ai, float* d_bi, float* d_ao, float* d_bo,
                     int L, int D, float* d_sc) {
    chunked_scan<Blelloch>(d_ai, d_bi, d_ao, d_bo, L, D, d_sc);
}

void launch_hillis_steele(float* d_ai, float* d_bi, float* d_ao, float* d_bo,
                          int L, int D, float* d_sc) {
    chunked_scan<HillisSteele>(d_ai, d_bi, d_ao, d_bo, L, D, d_sc);
}

KernelFn resolve_kernel(const std::string& kernel_name) {
    if (kernel_name == "warp_shuffle") return launch_warp_shuffle;
    if (kernel_name == "blelloch") return launch_blelloch;
    if (kernel_name == "hillis_steele") return launch_hillis_steele;
    return nullptr;
}

bool parse_int(const char* text, int* out) {
    if (!text || !out) return false;
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0') return false;
    if (value < 0 || value > 2147483647L) return false;
    *out = static_cast<int>(value);
    return true;
}

void print_usage(const char* argv0) {
    printf("Usage:\n");
    printf("  %s --kernel <warp_shuffle|blelloch|hillis_steele> --D <dim> --L <length> [options]\n", argv0);
    printf("\nOptions:\n");
    printf("  --input_dir <path>    Default: ../SyntheticData/inputs\n");
    printf("  --ref_dir <path>      Default: ../SequentialBaseline/SequentialData\n");
    printf("  --warmup <int>        Default: 3\n");
    printf("  --repeat <int>        Default: 10\n");
    printf("  --csv_append <path>   Append timing row to CSV file\n");
    printf("  --skip_check          Skip correctness check\n");
    printf("  --no_print            Suppress normal stdout output\n");
    printf("  --help                Show this message\n");
}

bool parse_args(int argc, char* argv[], Options* options) {
    if (!options) return false;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (std::strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            return false;
        } else if (std::strcmp(arg, "--kernel") == 0) {
            if (i + 1 >= argc) return false;
            options->kernel = argv[++i];
        } else if (std::strcmp(arg, "--D") == 0) {
            if (i + 1 >= argc) return false;
            if (!parse_int(argv[++i], &options->D)) return false;
        } else if (std::strcmp(arg, "--L") == 0) {
            if (i + 1 >= argc) return false;
            if (!parse_int(argv[++i], &options->L)) return false;
        } else if (std::strcmp(arg, "--input_dir") == 0) {
            if (i + 1 >= argc) return false;
            options->input_dir = argv[++i];
        } else if (std::strcmp(arg, "--ref_dir") == 0) {
            if (i + 1 >= argc) return false;
            options->ref_dir = argv[++i];
        } else if (std::strcmp(arg, "--warmup") == 0) {
            if (i + 1 >= argc) return false;
            if (!parse_int(argv[++i], &options->warmup)) return false;
        } else if (std::strcmp(arg, "--repeat") == 0) {
            if (i + 1 >= argc) return false;
            if (!parse_int(argv[++i], &options->repeat)) return false;
        } else if (std::strcmp(arg, "--csv_append") == 0) {
            if (i + 1 >= argc) return false;
            options->csv_append = argv[++i];
        } else if (std::strcmp(arg, "--no_print") == 0) {
            options->no_print = true;
        } else if (std::strcmp(arg, "--skip_check") == 0) {
            options->skip_check = true;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg);
            return false;
        }
    }

    if (options->kernel.empty() || options->D <= 0 || options->L <= 0 ||
        options->warmup < 0 || options->repeat <= 0) {
        return false;
    }
    return true;
}

bool file_exists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

bool load_inputs(const std::string& path, float* a, float* b, size_t n) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::perror(path.c_str());
        return false;
    }
    bool ok = std::fread(a, sizeof(float), n, f) == n
           && std::fread(b, sizeof(float), n, f) == n;
    std::fclose(f);
    return ok;
}

bool load_ref(const std::string& path, float* x, size_t n) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::perror(path.c_str());
        return false;
    }
    bool ok = std::fread(x, sizeof(float), n, f) == n;
    std::fclose(f);
    return ok;
}

void to_dim_major(const float* src, float* dst, int L, int D) {
    for (int t = 0; t < L; ++t)
        for (int d = 0; d < D; ++d)
            dst[d * L + t] = src[t * D + d];
}

bool check_output(const float* b_out_dm, const float* ref, int L, int D,
                  float tol = 1e-3f) {
    for (int t = 0; t < L; ++t) {
        for (int d = 0; d < D; ++d) {
            float got = b_out_dm[d * L + t];
            float expected = ref[t * D + d];
            float err = std::fabs(got - expected);
            float rel = err / fmaxf(std::fabs(expected), 1e-6f);
            if (err > tol && rel > tol) {
                fprintf(stderr,
                        "Mismatch kernel output: t=%d d=%d gpu=%.6f ref=%.6f abserr=%.2e relerr=%.2e\n",
                        t, d, got, expected, err, rel);
                return false;
            }
        }
    }
    return true;
}

float median(std::vector<float>* values) {
    if (!values || values->empty()) return -1.0f;
    std::sort(values->begin(), values->end());
    return (*values)[values->size() / 2];
}

bool append_csv_row(const std::string& csv_path,
                    const std::string& kernel,
                    int D,
                    int L,
                    float time_ms,
                    bool correct,
                    double throughput_gb_s) {
    bool need_header = !file_exists(csv_path);
    FILE* f = std::fopen(csv_path.c_str(), "a");
    if (!f) {
        std::perror(csv_path.c_str());
        return false;
    }
    if (need_header) {
        std::fprintf(f, "kernel,D,L,time_ms,correct,throughput_GB_s\n");
    }
    std::fprintf(f, "%s,%d,%d,%.6f,%d,%.6f\n",
                 kernel.c_str(), D, L, time_ms, correct ? 1 : 0, throughput_gb_s);
    std::fclose(f);
    return true;
}

bool run_benchmark(const Options& opt,
                   KernelFn kernel_fn,
                   const std::vector<float>& h_a_dm,
                   const std::vector<float>& h_b_dm,
                   const std::vector<float>& ref,
                   float* time_ms,
                   bool* correct,
                   double* throughput_gb_s) {
    if (!time_ms || !correct || !throughput_gb_s) return false;

    const int D = opt.D;
    const int L = opt.L;
    const size_t n = static_cast<size_t>(D) * L;

    float* d_ai = nullptr;
    float* d_bi = nullptr;
    float* d_ao = nullptr;
    float* d_bo = nullptr;
    float* d_sc = nullptr;

    CUDA_CHECK(cudaMalloc(&d_ai, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bi, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ao, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bo, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sc, 2ULL * D * L * sizeof(float)));

    for (int i = 0; i < opt.warmup; ++i) {
        CUDA_CHECK(cudaMemcpy(d_ai, h_a_dm.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bi, h_b_dm.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        kernel_fn(d_ai, d_bi, d_ao, d_bo, L, D, d_sc);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    *correct = true;
    if (!opt.skip_check) {
        if (opt.warmup == 0) {
            CUDA_CHECK(cudaMemcpy(d_ai, h_a_dm.data(), n * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_bi, h_b_dm.data(), n * sizeof(float), cudaMemcpyHostToDevice));
            kernel_fn(d_ai, d_bi, d_ao, d_bo, L, D, d_sc);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        std::vector<float> h_bo(n);
        CUDA_CHECK(cudaMemcpy(h_bo.data(), d_bo, n * sizeof(float), cudaMemcpyDeviceToHost));
        *correct = check_output(h_bo.data(), ref.data(), L, D);
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> samples;
    samples.reserve(opt.repeat);
    for (int r = 0; r < opt.repeat; ++r) {
        CUDA_CHECK(cudaMemcpy(d_ai, h_a_dm.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bi, h_b_dm.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(start));
        kernel_fn(d_ai, d_bi, d_ao, d_bo, L, D, d_sc);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        samples.push_back(elapsed);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    *time_ms = median(&samples);
    const double bytes = 3.0 * static_cast<double>(n) * sizeof(float);
    *throughput_gb_s = bytes / (*time_ms * 1e-3) * 1e-9;

    CUDA_CHECK(cudaFree(d_ai));
    CUDA_CHECK(cudaFree(d_bi));
    CUDA_CHECK(cudaFree(d_ao));
    CUDA_CHECK(cudaFree(d_bo));
    CUDA_CHECK(cudaFree(d_sc));
    return true;
}

}  // namespace

int main(int argc, char* argv[]) {
    Options opt;
    if (!parse_args(argc, argv, &opt)) {
        print_usage(argv[0]);
        return 1;
    }

    KernelFn kernel_fn = resolve_kernel(opt.kernel);
    if (!kernel_fn) {
        fprintf(stderr, "Unknown kernel '%s'\n", opt.kernel.c_str());
        return 1;
    }

    const int D = opt.D;
    const int L = opt.L;
    const size_t n = static_cast<size_t>(kBatchSize) * L * D;

    std::string input_path = opt.input_dir + "/input_B" + std::to_string(kBatchSize) +
                             "_L" + std::to_string(L) +
                             "_D" + std::to_string(D) + ".bin";
    std::string ref_path = opt.ref_dir + "/ref_B" + std::to_string(kBatchSize) +
                           "_L" + std::to_string(L) +
                           "_D" + std::to_string(D) + ".bin";

    std::vector<float> a_tm(n), b_tm(n), ref(n);
    if (!load_inputs(input_path, a_tm.data(), b_tm.data(), n)) {
        fprintf(stderr, "Failed to load inputs: %s\n", input_path.c_str());
        return 1;
    }
    if (!load_ref(ref_path, ref.data(), n)) {
        fprintf(stderr, "Failed to load reference: %s\n", ref_path.c_str());
        return 1;
    }

    std::vector<float> a_dm(n), b_dm(n);
    to_dim_major(a_tm.data(), a_dm.data(), L, D);
    to_dim_major(b_tm.data(), b_dm.data(), L, D);

    float time_ms = -1.0f;
    bool correct = false;
    double throughput_gb_s = 0.0;
    if (!run_benchmark(opt, kernel_fn, a_dm, b_dm, ref, &time_ms, &correct, &throughput_gb_s)) {
        return 1;
    }

    if (!opt.csv_append.empty()) {
        if (!append_csv_row(opt.csv_append, opt.kernel, D, L, time_ms, correct, throughput_gb_s)) {
            return 1;
        }
    }

    if (!opt.no_print) {
        printf("kernel=%-13s D=%-4d L=%-7d time_ms=%-10.4f GB/s=%-10.3f correct=%s\n",
               opt.kernel.c_str(), D, L, time_ms, throughput_gb_s, correct ? "PASS" : "FAIL");
    }

    return correct ? 0 : 2;
}
