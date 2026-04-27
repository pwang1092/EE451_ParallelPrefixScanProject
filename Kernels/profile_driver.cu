#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "warp_shuffle.cu"
#include "blelloch.cu"
#include "hillis_steele.cu"
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
    int L = -1;
    std::string input_dir = "../SyntheticData/inputs";
    std::string ref_dir = "../SequentialBaseline/SequentialData";
    std::string csv_append;
    int warmup = 3;
    int repeat = 10;
    bool no_print = false;
    bool skip_check = false;
};

using KernelFn = void (*)(Element*, Element*, int);

void launch_warp_shuffle(Element* d_in, Element* d_out, int L) {
    chunked_scan<WarpShuffle>(d_in, d_out, L);
}

void launch_blelloch(Element* d_in, Element* d_out, int L) {
    chunked_scan<Blelloch>(d_in, d_out, L);
}

void launch_hillis_steele(Element* d_in, Element* d_out, int L) {
    chunked_scan<HillisSteele>(d_in, d_out, L);
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
    printf("  %s --kernel <warp_shuffle|blelloch|hillis_steele> --L <length> [options]\n", argv0);
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

    if (options->kernel.empty() || options->L <= 0 || options->warmup < 0 || options->repeat <= 0) {
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

bool check_output(const Element* gpu_out, const float* ref, int L, float tol = 1e-3f) {
    for (int t = 0; t < L; ++t) {
        for (int d = 0; d < D; ++d) {
            float got = gpu_out[t].b[d];
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
                   const std::vector<Element>& h_in,
                   const std::vector<float>& ref,
                   float* time_ms,
                   bool* correct,
                   double* throughput_gb_s) {
    if (!time_ms || !correct || !throughput_gb_s) return false;

    const int L = opt.L;
    const size_t n = static_cast<size_t>(L) * D;

    Element* d_in = nullptr;
    Element* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, L * sizeof(Element)));
    CUDA_CHECK(cudaMalloc(&d_out, L * sizeof(Element)));

    for (int i = 0; i < opt.warmup; ++i) {
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), L * sizeof(Element), cudaMemcpyHostToDevice));
        kernel_fn(d_in, d_out, L);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    *correct = true;
    if (!opt.skip_check) {
        if (opt.warmup == 0) {
            CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), L * sizeof(Element), cudaMemcpyHostToDevice));
            kernel_fn(d_in, d_out, L);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        std::vector<Element> h_out(L);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, L * sizeof(Element), cudaMemcpyDeviceToHost));
        *correct = check_output(h_out.data(), ref.data(), L);
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> samples;
    samples.reserve(opt.repeat);
    for (int r = 0; r < opt.repeat; ++r) {
        CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), L * sizeof(Element), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(start));
        kernel_fn(d_in, d_out, L);
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

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
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

    const int L = opt.L;
    const size_t n = static_cast<size_t>(kBatchSize) * L * D;

    std::string input_path = opt.input_dir + "/input_B" + std::to_string(kBatchSize) +
                             "_L" + std::to_string(L) +
                             "_D" + std::to_string(D) + ".bin";
    std::string ref_path = opt.ref_dir + "/ref_B" + std::to_string(kBatchSize) +
                           "_L" + std::to_string(L) +
                           "_D" + std::to_string(D) + ".bin";

    std::vector<float> a(n), b(n), ref(n);
    if (!load_inputs(input_path, a.data(), b.data(), n)) {
        fprintf(stderr, "Failed to load inputs: %s\n", input_path.c_str());
        return 1;
    }
    if (!load_ref(ref_path, ref.data(), n)) {
        fprintf(stderr, "Failed to load reference: %s\n", ref_path.c_str());
        return 1;
    }

    std::vector<Element> h_in(L);
    for (int t = 0; t < L; ++t) {
        for (int d = 0; d < D; ++d) {
            h_in[t].a[d] = a[t * D + d];
            h_in[t].b[d] = b[t * D + d];
        }
    }

    float time_ms = -1.0f;
    bool correct = false;
    double throughput_gb_s = 0.0;
    if (!run_benchmark(opt, kernel_fn, h_in, ref, &time_ms, &correct, &throughput_gb_s)) {
        return 1;
    }

    if (!opt.csv_append.empty()) {
        if (!append_csv_row(opt.csv_append, opt.kernel, L, time_ms, correct, throughput_gb_s)) {
            return 1;
        }
    }

    if (!opt.no_print) {
        printf("kernel=%-13s D=%-4d L=%-7d time_ms=%-10.4f GB/s=%-10.3f correct=%s\n",
               opt.kernel.c_str(), D, L, time_ms, throughput_gb_s, correct ? "PASS" : "FAIL");
    }

    return correct ? 0 : 2;
}
