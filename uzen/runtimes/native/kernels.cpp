// Native C++ kernels for UZEN's NIR executor.
//
// These are real compute kernels with a C ABI, loaded through ctypes. The
// Python layer validates shapes and dtypes before calling them; the kernels
// still refuse null pointers and non-positive dimensions. Accumulation for
// normalization, activations, softmax and attention uses double precision to
// match UZEN's NumPy reference, while storage stays float32.
//
// Build with:
//   g++ -O3 -std=c++17 -fPIC -shared -fopenmp kernels.cpp -o libuzen_native.so
//
// OpenMP is optional. Without -fopenmp the pragmas are ignored and the code
// remains correct, only single-threaded.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

inline bool valid(const void* pointer) { return pointer != nullptr; }

inline double gelu_tanh(double x) {
    const double c = std::sqrt(2.0 / kPi);
    return 0.5 * x * (1.0 + std::tanh(c * (x + 0.044715 * x * x * x)));
}

inline double gelu_exact(double x) {
    return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

inline double silu(double x) { return x / (1.0 + std::exp(-x)); }

}  // namespace

extern "C" {

int uzen_linear_f32(const float* x, const float* w, const float* bias, float* out,
                    int rows, int in_features, int out_features) {
    if (!valid(x) || !valid(w) || !valid(out) || rows < 0 || in_features <= 0 ||
        out_features <= 0) {
        return 1;
    }
    if (rows == 0) {
        return 0;
    }
#pragma omp parallel for schedule(static) if (rows > 1 && rows * out_features > 256)
    for (int row = 0; row < rows; ++row) {
        const float* xrow = x + static_cast<std::size_t>(row) * in_features;
        float* orow = out + static_cast<std::size_t>(row) * out_features;
        for (int o = 0; o < out_features; ++o) {
            const float* wrow = w + static_cast<std::size_t>(o) * in_features;
            float acc = 0.0f;
            for (int k = 0; k < in_features; ++k) {
                acc += xrow[k] * wrow[k];
            }
            orow[o] = acc + (bias ? bias[o] : 0.0f);
        }
    }
    return 0;
}

int uzen_matmul_f32(const float* a, const float* b, float* out, int m, int k, int n) {
    if (!valid(a) || !valid(b) || !valid(out) || m < 0 || k <= 0 || n <= 0) {
        return 1;
    }
    if (m == 0) {
        return 0;
    }
#pragma omp parallel for schedule(static) if (m > 1 && static_cast<long long>(m) * n > 256)
    for (int i = 0; i < m; ++i) {
        float* orow = out + static_cast<std::size_t>(i) * n;
        for (int j = 0; j < n; ++j) {
            orow[j] = 0.0f;
        }
        for (int p = 0; p < k; ++p) {
            const float aip = a[static_cast<std::size_t>(i) * k + p];
            const float* brow = b + static_cast<std::size_t>(p) * n;
            for (int j = 0; j < n; ++j) {
                orow[j] += aip * brow[j];
            }
        }
    }
    return 0;
}

int uzen_layernorm_f32(const float* x, const float* weight, const float* bias, float* out,
                       int rows, int cols, float eps) {
    if (!valid(x) || !valid(out) || rows < 0 || cols <= 0 || eps <= 0.0f) {
        return 1;
    }
    if (rows == 0) {
        return 0;
    }
    const double epsilon = static_cast<double>(eps);
#pragma omp parallel for schedule(static) if (rows > 1)
    for (int row = 0; row < rows; ++row) {
        const float* xrow = x + static_cast<std::size_t>(row) * cols;
        float* orow = out + static_cast<std::size_t>(row) * cols;
        double sum = 0.0;
        for (int c = 0; c < cols; ++c) {
            sum += xrow[c];
        }
        const double mean = sum / cols;
        double variance = 0.0;
        for (int c = 0; c < cols; ++c) {
            const double delta = static_cast<double>(xrow[c]) - mean;
            variance += delta * delta;
        }
        variance /= cols;
        const double inverse = 1.0 / std::sqrt(variance + epsilon);
        for (int c = 0; c < cols; ++c) {
            double value = (static_cast<double>(xrow[c]) - mean) * inverse;
            if (weight) {
                value *= static_cast<double>(weight[c]);
            }
            if (bias) {
                value += static_cast<double>(bias[c]);
            }
            orow[c] = static_cast<float>(value);
        }
    }
    return 0;
}

int uzen_rmsnorm_f32(const float* x, const float* weight, float* out, int rows, int cols,
                     float eps) {
    if (!valid(x) || !valid(out) || rows < 0 || cols <= 0 || eps <= 0.0f) {
        return 1;
    }
    if (rows == 0) {
        return 0;
    }
    const double epsilon = static_cast<double>(eps);
#pragma omp parallel for schedule(static) if (rows > 1)
    for (int row = 0; row < rows; ++row) {
        const float* xrow = x + static_cast<std::size_t>(row) * cols;
        float* orow = out + static_cast<std::size_t>(row) * cols;
        double squares = 0.0;
        for (int c = 0; c < cols; ++c) {
            const double value = xrow[c];
            squares += value * value;
        }
        const double inverse = 1.0 / std::sqrt(squares / cols + epsilon);
        for (int c = 0; c < cols; ++c) {
            double value = static_cast<double>(xrow[c]) * inverse;
            if (weight) {
                value *= static_cast<double>(weight[c]);
            }
            orow[c] = static_cast<float>(value);
        }
    }
    return 0;
}

int uzen_softmax_f32(const float* x, float* out, int rows, int cols) {
    if (!valid(x) || !valid(out) || rows < 0 || cols <= 0) {
        return 1;
    }
    if (rows == 0) {
        return 0;
    }
#pragma omp parallel for schedule(static) if (rows > 1)
    for (int row = 0; row < rows; ++row) {
        const float* xrow = x + static_cast<std::size_t>(row) * cols;
        float* orow = out + static_cast<std::size_t>(row) * cols;
        double maximum = static_cast<double>(xrow[0]);
        for (int c = 1; c < cols; ++c) {
            maximum = std::max(maximum, static_cast<double>(xrow[c]));
        }
        double total = 0.0;
        for (int c = 0; c < cols; ++c) {
            const double value = std::exp(static_cast<double>(xrow[c]) - maximum);
            orow[c] = static_cast<float>(value);
            total += value;
        }
        if (total > 0.0) {
            for (int c = 0; c < cols; ++c) {
                orow[c] = static_cast<float>(static_cast<double>(orow[c]) / total);
            }
        }
    }
    return 0;
}

#define UZEN_UNARY_KERNEL(name, expression)                                        \
    int name(const float* x, float* out, int count) {                             \
        if (!valid(x) || !valid(out) || count < 0) {                              \
            return 1;                                                              \
        }                                                                          \
        for (int i = 0; i < count; ++i) {                                          \
            const double value = static_cast<double>(x[i]);                        \
            out[i] = static_cast<float>(expression);                               \
        }                                                                          \
        return 0;                                                                  \
    }

UZEN_UNARY_KERNEL(uzen_gelu_f32, gelu_tanh(value))
UZEN_UNARY_KERNEL(uzen_gelu_exact_f32, gelu_exact(value))
UZEN_UNARY_KERNEL(uzen_silu_f32, silu(value))
UZEN_UNARY_KERNEL(uzen_tanh_f32, std::tanh(value))
UZEN_UNARY_KERNEL(uzen_sigmoid_f32, 1.0 / (1.0 + std::exp(-value)))
UZEN_UNARY_KERNEL(uzen_relu_f32, value > 0.0 ? value : 0.0)

#undef UZEN_UNARY_KERNEL

int uzen_add_f32(const float* a, const float* b, float* out, int count) {
    if (!valid(a) || !valid(b) || !valid(out) || count < 0) {
        return 1;
    }
    for (int i = 0; i < count; ++i) {
        out[i] = a[i] + b[i];
    }
    return 0;
}

int uzen_sub_f32(const float* a, const float* b, float* out, int count) {
    if (!valid(a) || !valid(b) || !valid(out) || count < 0) {
        return 1;
    }
    for (int i = 0; i < count; ++i) {
        out[i] = a[i] - b[i];
    }
    return 0;
}

int uzen_mul_f32(const float* a, const float* b, float* out, int count) {
    if (!valid(a) || !valid(b) || !valid(out) || count < 0) {
        return 1;
    }
    for (int i = 0; i < count; ++i) {
        out[i] = a[i] * b[i];
    }
    return 0;
}

int uzen_scale_f32(const float* x, float* out, int count, float factor) {
    if (!valid(x) || !valid(out) || count < 0) {
        return 1;
    }
    for (int i = 0; i < count; ++i) {
        out[i] = x[i] * factor;
    }
    return 0;
}

int uzen_clip_f32(const float* x, float* out, int count, float lower, float upper) {
    if (!valid(x) || !valid(out) || count < 0 || lower > upper) {
        return 1;
    }
    for (int i = 0; i < count; ++i) {
        out[i] = std::min(std::max(x[i], lower), upper);
    }
    return 0;
}

int uzen_embedding_f32(const std::int64_t* indices, const float* weight, float* out,
                       std::int64_t count, int dimensions) {
    if (!valid(indices) || !valid(weight) || !valid(out) || count < 0 || dimensions <= 0) {
        return 1;
    }
    for (std::int64_t i = 0; i < count; ++i) {
        const std::int64_t index = indices[i];
        const float* wrow = weight + static_cast<std::size_t>(index) * dimensions;
        float* orow = out + static_cast<std::size_t>(i) * dimensions;
        for (int d = 0; d < dimensions; ++d) {
            orow[d] = wrow[d];
        }
    }
    return 0;
}

int uzen_reduce_last_f32(const float* x, float* out, int rows, int cols, int mode) {
    if (!valid(x) || !valid(out) || rows < 0 || cols <= 0) {
        return 1;
    }
    for (int row = 0; row < rows; ++row) {
        const float* xrow = x + static_cast<std::size_t>(row) * cols;
        double total = 0.0;
        for (int c = 0; c < cols; ++c) {
            total += xrow[c];
        }
        out[row] = static_cast<float>(mode == 0 ? total / cols : total);
    }
    return 0;
}

int uzen_attention_f32(const float* q, const float* k, const float* v, float* out,
                       int batch, int heads, int q_seq, int kv_seq, int dimensions,
                       float scale, int causal, const float* alibi, int window) {
    if (!valid(q) || !valid(k) || !valid(v) || !valid(out) || batch <= 0 || heads <= 0 ||
        q_seq < 0 || kv_seq <= 0 || dimensions <= 0) {
        return 1;
    }
    if (q_seq == 0) {
        return 0;
    }
    const int q_offset = kv_seq - q_seq;
#pragma omp parallel for collapse(2) schedule(static) if (batch * heads > 1)
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int t = 0; t < q_seq; ++t) {
                const std::size_t qbase =
                    ((static_cast<std::size_t>(b) * heads + h) * q_seq + t) * dimensions;
                const float* qrow = q + qbase;
                const int qpos = q_offset + t;
                double maximum = -std::numeric_limits<double>::infinity();
                for (int s = 0; s < kv_seq; ++s) {
                    if (causal && s > qpos) {
                        continue;
                    }
                    if (window >= 0 && std::abs(qpos - s) > window) {
                        continue;
                    }
                    const std::size_t kbase =
                        ((static_cast<std::size_t>(b) * heads + h) * kv_seq + s) * dimensions;
                    const float* krow = k + kbase;
                    double dot = 0.0;
                    for (int d = 0; d < dimensions; ++d) {
                        dot += static_cast<double>(qrow[d]) * static_cast<double>(krow[d]);
                    }
                    double score = dot * static_cast<double>(scale);
                    if (alibi) {
                        const double distance =
                            causal ? static_cast<double>(qpos - s) : std::abs(qpos - s);
                        score -= static_cast<double>(alibi[h]) * distance;
                    }
                    maximum = std::max(maximum, score);
                }
                if (!std::isfinite(maximum)) {
                    maximum = 0.0;
                }
                std::vector<double> accumulator(static_cast<std::size_t>(dimensions), 0.0);
                double denominator = 0.0;
                for (int s = 0; s < kv_seq; ++s) {
                    if (causal && s > qpos) {
                        continue;
                    }
                    if (window >= 0 && std::abs(qpos - s) > window) {
                        continue;
                    }
                    const std::size_t kbase =
                        ((static_cast<std::size_t>(b) * heads + h) * kv_seq + s) * dimensions;
                    const float* krow = k + kbase;
                    double dot = 0.0;
                    for (int d = 0; d < dimensions; ++d) {
                        dot += static_cast<double>(qrow[d]) * static_cast<double>(krow[d]);
                    }
                    double score = dot * static_cast<double>(scale);
                    if (alibi) {
                        const double distance =
                            causal ? static_cast<double>(qpos - s) : std::abs(qpos - s);
                        score -= static_cast<double>(alibi[h]) * distance;
                    }
                    const double weight = std::exp(score - maximum);
                    denominator += weight;
                    const std::size_t vbase =
                        ((static_cast<std::size_t>(b) * heads + h) * kv_seq + s) * dimensions;
                    const float* vrow = v + vbase;
                    for (int d = 0; d < dimensions; ++d) {
                        accumulator[d] += weight * static_cast<double>(vrow[d]);
                    }
                }
                float* orow = out + qbase;
                if (denominator <= 0.0) {
                    for (int d = 0; d < dimensions; ++d) {
                        orow[d] = 0.0f;
                    }
                } else {
                    for (int d = 0; d < dimensions; ++d) {
                        orow[d] = static_cast<float>(accumulator[d] / denominator);
                    }
                }
            }
        }
    }
    return 0;
}

}  // extern "C"
