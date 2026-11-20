// Compiled-path kernels for UZEN's native execution plans (phase N2).
//
// kernels.cpp holds the direct executor's kernels (one call per NIR node). This
// translation unit adds what a compiled plan needs:
//
//   * prepacked, cache-blocked GEMM with an optional fused activation,
//   * a fused residual-add + normalization kernel,
//   * convolution, pooling, recurrent (RNN/LSTM/GRU), state-space, graph
//     convolution, sparse mixture-of-experts and int8 quantized linear kernels,
//   * a single-pass (flash-style) attention kernel that never materializes the
//     score matrix.
//
// Accumulation uses double precision where the NumPy reference does, storage is
// float32. Every kernel refuses null pointers and non-positive dimensions.
//
// Build with kernels.cpp; both files are compiled into one shared library.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr int kBlockN = 64;  // output-feature block (contiguous in packed form)
constexpr int kBlockK = 64;  // reduction block
constexpr double kPi = 3.14159265358979323846;

inline bool valid(const void* pointer) { return pointer != nullptr; }

inline double activate(double value, int code) {
    switch (code) {
        case 1:  // relu
            return value > 0.0 ? value : 0.0;
        case 2: {  // gelu (tanh approximation)
            const double c = std::sqrt(2.0 / kPi);
            return 0.5 * value * (1.0 + std::tanh(c * (value + 0.044715 * value * value * value)));
        }
        case 3:  // silu
            return value / (1.0 + std::exp(-value));
        case 4:  // tanh
            return std::tanh(value);
        case 5:  // sigmoid
            return 1.0 / (1.0 + std::exp(-value));
        case 6:  // gelu (exact erf)
            return 0.5 * value * (1.0 + std::erf(value / std::sqrt(2.0)));
        default:
            return value;
    }
}

inline double sigmoid(double value) { return 1.0 / (1.0 + std::exp(-value)); }

}  // namespace

extern "C" {

// ------------------------------------------------------------------------- //
// prepacked GEMM
// ------------------------------------------------------------------------- //

// Pack (out_features, in_features) row-major weights into the blocked layout the
// GEMM consumes: for each output block nb and reduction block kb, a contiguous
// kBlockK x kBlockN panel. Partial blocks are zero-padded, which removes bounds
// checks from the inner loop.
int uzen_pack_weight_f32(const float* w, float* packed, int out_features, int in_features) {
    if (!valid(w) || !valid(packed) || out_features <= 0 || in_features <= 0) {
        return 1;
    }
    const int nblocks = (out_features + kBlockN - 1) / kBlockN;
    const int kblocks = (in_features + kBlockK - 1) / kBlockK;
    for (int nb = 0; nb < nblocks; ++nb) {
        for (int kb = 0; kb < kblocks; ++kb) {
            float* panel = packed + (static_cast<std::size_t>(nb) * kblocks + kb) * kBlockK * kBlockN;
            for (int kk = 0; kk < kBlockK; ++kk) {
                const int k = kb * kBlockK + kk;
                float* prow = panel + static_cast<std::size_t>(kk) * kBlockN;
                for (int nn = 0; nn < kBlockN; ++nn) {
                    const int o = nb * kBlockN + nn;
                    prow[nn] = (k < in_features && o < out_features)
                                   ? w[static_cast<std::size_t>(o) * in_features + k]
                                   : 0.0f;
                }
            }
        }
    }
    return 0;
}

int uzen_packed_size(int out_features, int in_features) {
    if (out_features <= 0 || in_features <= 0) {
        return 0;
    }
    const int nblocks = (out_features + kBlockN - 1) / kBlockN;
    const int kblocks = (in_features + kBlockK - 1) / kBlockK;
    return nblocks * kblocks * kBlockK * kBlockN;
}

// y = activation(x @ W^T + b) with W in packed form. Rows are independent, so the
// parallelism is over rows and each row streams contiguous panels of W.
int uzen_gemm_packed_f32(const float* x, const float* packed, const float* bias, float* out,
                         int rows, int in_features, int out_features, int activation) {
    if (!valid(x) || !valid(packed) || !valid(out) || rows < 0 || in_features <= 0 ||
        out_features <= 0) {
        return 1;
    }
    if (rows == 0) {
        return 0;
    }
    const int nblocks = (out_features + kBlockN - 1) / kBlockN;
    const int kblocks = (in_features + kBlockK - 1) / kBlockK;
#pragma omp parallel for schedule(static) if (rows > 1)
    for (int row = 0; row < rows; ++row) {
        const float* xrow = x + static_cast<std::size_t>(row) * in_features;
        float* orow = out + static_cast<std::size_t>(row) * out_features;
        for (int nb = 0; nb < nblocks; ++nb) {
            double acc[kBlockN];
            for (int nn = 0; nn < kBlockN; ++nn) {
                acc[nn] = 0.0;
            }
            for (int kb = 0; kb < kblocks; ++kb) {
                const float* panel =
                    packed + (static_cast<std::size_t>(nb) * kblocks + kb) * kBlockK * kBlockN;
                const float* xblock = xrow + static_cast<std::size_t>(kb) * kBlockK;
                const int kcount =
                    std::min(kBlockK, in_features - kb * kBlockK);
                for (int kk = 0; kk < kcount; ++kk) {
                    const double value = xblock[kk];
                    const float* prow = panel + static_cast<std::size_t>(kk) * kBlockN;
                    for (int nn = 0; nn < kBlockN; ++nn) {
                        acc[nn] += value * static_cast<double>(prow[nn]);
                    }
                }
            }
            const int base = nb * kBlockN;
            const int count = std::min(kBlockN, out_features - base);
            for (int nn = 0; nn < count; ++nn) {
                const int o = base + nn;
                double value = acc[nn];
                if (bias) {
                    value += static_cast<double>(bias[o]);
                }
                orow[o] = static_cast<float>(activate(value, activation));
            }
        }
    }
    return 0;
}

// ------------------------------------------------------------------------- //
// fused residual add + normalization
// ------------------------------------------------------------------------- //

// value = a + b, optionally stored to `residual`, then normalized. mode 0 is
// layernorm (mean/variance) and mode 1 is rmsnorm.
int uzen_add_norm_f32(const float* a, const float* b, const float* weight, const float* bias,
                      float* out, float* residual, int rows, int cols, float eps, int mode) {
    if (!valid(a) || !valid(b) || !valid(out) || rows < 0 || cols <= 0 || eps <= 0.0f) {
        return 1;
    }
    if (mode != 0 && mode != 1) {
        return 2;
    }
    if (rows == 0) {
        return 0;
    }
    const double epsilon = static_cast<double>(eps);
#pragma omp parallel for schedule(static) if (rows > 1)
    for (int row = 0; row < rows; ++row) {
        const float* arow = a + static_cast<std::size_t>(row) * cols;
        const float* brow = b + static_cast<std::size_t>(row) * cols;
        float* orow = out + static_cast<std::size_t>(row) * cols;
        if (residual) {
            float* rrow = residual + static_cast<std::size_t>(row) * cols;
            for (int c = 0; c < cols; ++c) {
                rrow[c] = arow[c] + brow[c];
            }
        }
        if (mode == 0) {
            double sum = 0.0;
            for (int c = 0; c < cols; ++c) {
                sum += static_cast<double>(arow[c]) + static_cast<double>(brow[c]);
            }
            const double mean = sum / cols;
            double variance = 0.0;
            for (int c = 0; c < cols; ++c) {
                const double delta =
                    (static_cast<double>(arow[c]) + static_cast<double>(brow[c])) - mean;
                variance += delta * delta;
            }
            variance /= cols;
            const double inverse = 1.0 / std::sqrt(variance + epsilon);
            for (int c = 0; c < cols; ++c) {
                double value =
                    ((static_cast<double>(arow[c]) + static_cast<double>(brow[c])) - mean) *
                    inverse;
                if (weight) {
                    value *= static_cast<double>(weight[c]);
                }
                if (bias) {
                    value += static_cast<double>(bias[c]);
                }
                orow[c] = static_cast<float>(value);
            }
        } else {
            double squares = 0.0;
            for (int c = 0; c < cols; ++c) {
                const double value = static_cast<double>(arow[c]) + static_cast<double>(brow[c]);
                squares += value * value;
            }
            const double inverse = 1.0 / std::sqrt(squares / cols + epsilon);
            for (int c = 0; c < cols; ++c) {
                double value =
                    (static_cast<double>(arow[c]) + static_cast<double>(brow[c])) * inverse;
                if (weight) {
                    value *= static_cast<double>(weight[c]);
                }
                orow[c] = static_cast<float>(value);
            }
        }
    }
    return 0;
}

// ------------------------------------------------------------------------- //
// convolution and pooling
// ------------------------------------------------------------------------- //

// Direct (im2col-free) cross-correlation over NCHW input and (O, C, kh, kw)
// weights. Padding is zero for convolution.
int uzen_conv2d_f32(const float* x, const float* w, const float* bias, float* out, int batch,
                    int in_channels, int in_h, int in_w, int out_channels, int kh, int kw,
                    int stride_h, int stride_w, int pad_h, int pad_w, int out_h, int out_w) {
    if (!valid(x) || !valid(w) || !valid(out) || batch <= 0 || in_channels <= 0 || in_h <= 0 ||
        in_w <= 0 || out_channels <= 0 || kh <= 0 || kw <= 0 || stride_h <= 0 || stride_w <= 0 ||
        pad_h < 0 || pad_w < 0 || out_h <= 0 || out_w <= 0) {
        return 1;
    }
#pragma omp parallel for collapse(2) schedule(static) if (batch * out_channels > 1)
    for (int b = 0; b < batch; ++b) {
        for (int o = 0; o < out_channels; ++o) {
            const float* wbase = w + static_cast<std::size_t>(o) * in_channels * kh * kw;
            float* obase = out + ((static_cast<std::size_t>(b) * out_channels + o) * out_h) * out_w;
            const double bias_value = bias ? static_cast<double>(bias[o]) : 0.0;
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    double acc = bias_value;
                    for (int c = 0; c < in_channels; ++c) {
                        const float* xbase =
                            x + ((static_cast<std::size_t>(b) * in_channels + c) * in_h) * in_w;
                        const float* wchan = wbase + static_cast<std::size_t>(c) * kh * kw;
                        for (int i = 0; i < kh; ++i) {
                            const int ih = oh * stride_h - pad_h + i;
                            if (ih < 0 || ih >= in_h) {
                                continue;
                            }
                            for (int j = 0; j < kw; ++j) {
                                const int iw = ow * stride_w - pad_w + j;
                                if (iw < 0 || iw >= in_w) {
                                    continue;
                                }
                                acc += static_cast<double>(xbase[static_cast<std::size_t>(ih) * in_w + iw]) *
                                       static_cast<double>(wchan[static_cast<std::size_t>(i) * kw + j]);
                            }
                        }
                    }
                    obase[static_cast<std::size_t>(oh) * out_w + ow] = static_cast<float>(acc);
                }
            }
        }
    }
    return 0;
}

// Pooling over NCHW. mode 0 is max (padding contributes -inf, matching the NumPy
// reference), mode 1 is average over the padded window.
int uzen_pool2d_f32(const float* x, float* out, int batch, int channels, int in_h, int in_w,
                    int kh, int kw, int stride_h, int stride_w, int pad_h, int pad_w, int out_h,
                    int out_w, int mode) {
    if (!valid(x) || !valid(out) || batch <= 0 || channels <= 0 || in_h <= 0 || in_w <= 0 ||
        kh <= 0 || kw <= 0 || stride_h <= 0 || stride_w <= 0 || pad_h < 0 || pad_w < 0 ||
        out_h <= 0 || out_w <= 0 || (mode != 0 && mode != 1)) {
        return 1;
    }
#pragma omp parallel for collapse(2) schedule(static) if (batch * channels > 1)
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            const float* xbase =
                x + ((static_cast<std::size_t>(b) * channels + c) * in_h) * in_w;
            float* obase = out + ((static_cast<std::size_t>(b) * channels + c) * out_h) * out_w;
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    double best = -std::numeric_limits<double>::infinity();
                    double total = 0.0;
                    for (int i = 0; i < kh; ++i) {
                        const int ih = oh * stride_h - pad_h + i;
                        for (int j = 0; j < kw; ++j) {
                            const int iw = ow * stride_w - pad_w + j;
                            double value = -std::numeric_limits<double>::infinity();
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                value = static_cast<double>(
                                    xbase[static_cast<std::size_t>(ih) * in_w + iw]);
                            }
                            best = std::max(best, value);
                            total += value;
                        }
                    }
                    const double result = (mode == 0) ? best : total / (kh * kw);
                    obase[static_cast<std::size_t>(oh) * out_w + ow] = static_cast<float>(result);
                }
            }
        }
    }
    return 0;
}

// ------------------------------------------------------------------------- //
// recurrent families
// ------------------------------------------------------------------------- //

// kind: 0 = tanh RNN, 1 = LSTM (i, f, g, o gate order), 2 = GRU (r, z, n).
// w_ih is (gates*hidden, features), w_hh is (gates*hidden, hidden).
int uzen_recurrent_f32(const float* x, const float* w_ih, const float* w_hh, const float* bias,
                       float* out, int batch, int steps, int features, int hidden, int kind) {
    if (!valid(x) || !valid(w_ih) || !valid(w_hh) || !valid(out) || batch <= 0 || steps <= 0 ||
        features <= 0 || hidden <= 0 || (kind < 0 || kind > 2)) {
        return 1;
    }
    const int gates = (kind == 1) ? 4 : (kind == 2) ? 3 : 1;
    const int width = gates * hidden;
#pragma omp parallel for schedule(static) if (batch > 1)
    for (int b = 0; b < batch; ++b) {
        std::vector<double> h(static_cast<std::size_t>(hidden), 0.0);
        std::vector<double> c(static_cast<std::size_t>(hidden), 0.0);
        std::vector<double> gate(static_cast<std::size_t>(width), 0.0);
        std::vector<double> hidden_gate(static_cast<std::size_t>(width), 0.0);
        for (int t = 0; t < steps; ++t) {
            const float* xt =
                x + ((static_cast<std::size_t>(b) * steps + t) * features);
            for (int g = 0; g < width; ++g) {
                double acc = bias ? static_cast<double>(bias[g]) : 0.0;
                const float* wrow = w_ih + static_cast<std::size_t>(g) * features;
                for (int f = 0; f < features; ++f) {
                    acc += static_cast<double>(xt[f]) * static_cast<double>(wrow[f]);
                }
                gate[g] = acc;
            }
            for (int g = 0; g < width; ++g) {
                double acc = 0.0;
                const float* wrow = w_hh + static_cast<std::size_t>(g) * hidden;
                for (int f = 0; f < hidden; ++f) {
                    acc += h[f] * static_cast<double>(wrow[f]);
                }
                hidden_gate[g] = acc;
            }
            float* ot = out + ((static_cast<std::size_t>(b) * steps + t) * hidden);
            if (kind == 0) {
                // h = tanh(x + h_prev)
                for (int i = 0; i < hidden; ++i) {
                    h[i] = std::tanh(gate[i] + hidden_gate[i]);
                    ot[i] = static_cast<float>(h[i]);
                }
            } else if (kind == 1) {
                // Every LSTM gate is a function of x_t + h_{t-1}.
                for (int i = 0; i < hidden; ++i) {
                    const double input_gate = sigmoid(gate[i] + hidden_gate[i]);
                    const double forget_gate =
                        sigmoid(gate[hidden + i] + hidden_gate[hidden + i]);
                    const double cell_gate =
                        std::tanh(gate[2 * hidden + i] + hidden_gate[2 * hidden + i]);
                    const double output_gate =
                        sigmoid(gate[3 * hidden + i] + hidden_gate[3 * hidden + i]);
                    c[i] = forget_gate * c[i] + input_gate * cell_gate;
                    h[i] = output_gate * std::tanh(c[i]);
                    ot[i] = static_cast<float>(h[i]);
                }
            } else {
                // Reset/update gates see x + h; the candidate sees x + r * h only.
                for (int i = 0; i < hidden; ++i) {
                    const double reset = sigmoid(gate[i] + hidden_gate[i]);
                    const double update = sigmoid(gate[hidden + i] + hidden_gate[hidden + i]);
                    const double candidate =
                        std::tanh(gate[2 * hidden + i] + reset * hidden_gate[2 * hidden + i]);
                    h[i] = (1.0 - update) * candidate + update * h[i];
                    ot[i] = static_cast<float>(h[i]);
                }
            }
        }
    }
    return 0;
}

// Diagonal state-space recurrence:
//   h_t = A * h_{t-1} + B * x_t ; y_t = <C, h_t> + D * x_t
int uzen_ssm_f32(const float* x, const float* a, const float* b, const float* c, const float* d,
                 float* out, int batch, int steps, int channels, int state) {
    if (!valid(x) || !valid(a) || !valid(b) || !valid(c) || !valid(out) || batch <= 0 ||
        steps <= 0 || channels <= 0 || state <= 0) {
        return 1;
    }
#pragma omp parallel for schedule(static) if (batch > 1)
    for (int n = 0; n < batch; ++n) {
        std::vector<double> h(static_cast<std::size_t>(channels) * state, 0.0);
        for (int t = 0; t < steps; ++t) {
            const float* xt = x + ((static_cast<std::size_t>(n) * steps + t) * channels);
            for (int ch = 0; ch < channels; ++ch) {
                const double value = static_cast<double>(xt[ch]);
                const float* arow = a + static_cast<std::size_t>(ch) * state;
                const float* brow = b + static_cast<std::size_t>(ch) * state;
                const float* crow = c + static_cast<std::size_t>(ch) * state;
                double* hrow = h.data() + static_cast<std::size_t>(ch) * state;
                double y = d ? static_cast<double>(d[ch]) * value : 0.0;
                for (int s = 0; s < state; ++s) {
                    hrow[s] = static_cast<double>(arow[s]) * hrow[s] +
                              static_cast<double>(brow[s]) * value;
                    y += static_cast<double>(crow[s]) * hrow[s];
                }
                out[((static_cast<std::size_t>(n) * steps + t) * channels) + ch] =
                    static_cast<float>(y);
            }
        }
    }
    return 0;
}

// ------------------------------------------------------------------------- //
// graph convolution and sparse mixture of experts
// ------------------------------------------------------------------------- //

// y = A_norm @ X @ W^T + b, where A_norm = (A + I) scaled by 1/sqrt(deg) on both
// sides (degree measured after adding self loops).
int uzen_gcn_f32(const float* x, const float* adj, const float* w, const float* bias, float* out,
                 int batch, int nodes, int in_features, int out_features) {
    if (!valid(x) || !valid(adj) || !valid(w) || !valid(out) || batch <= 0 || nodes <= 0 ||
        in_features <= 0 || out_features <= 0) {
        return 1;
    }
#pragma omp parallel
    {
        std::vector<double> inverse(static_cast<std::size_t>(nodes));
#pragma omp for schedule(static)
        for (int n = 0; n < batch; ++n) {
            // inv[j] = 1 / sqrt(deg_j) over the self-looped adjacency, and the
            // reference scales entry (i, j) by inv[i] * inv[j].
            for (int j = 0; j < nodes; ++j) {
                const float* rowj =
                    adj + (static_cast<std::size_t>(n) * nodes + j) * nodes;
                double degree = 1.0;  // self loop
                for (int k = 0; k < nodes; ++k) {
                    degree += static_cast<double>(rowj[k]);
                }
                inverse[j] = 1.0 / std::sqrt(std::max(degree, 1e-12));
            }
            for (int i = 0; i < nodes; ++i) {
                const float* arow =
                    adj + (static_cast<std::size_t>(n) * nodes + i) * nodes;
                std::vector<double> aggregated(static_cast<std::size_t>(in_features), 0.0);
                for (int j = 0; j < nodes; ++j) {
                    const double weight =
                        ((i == j ? 1.0 : 0.0) + static_cast<double>(arow[j])) *
                        inverse[i] * inverse[j];
                    const float* xrow =
                        x + ((static_cast<std::size_t>(n) * nodes + j) * in_features);
                    for (int f = 0; f < in_features; ++f) {
                        aggregated[f] += weight * static_cast<double>(xrow[f]);
                    }
                }
                float* orow =
                    out + ((static_cast<std::size_t>(n) * nodes + i) * out_features);
                for (int o = 0; o < out_features; ++o) {
                    const float* wrow = w + static_cast<std::size_t>(o) * in_features;
                    double acc = bias ? static_cast<double>(bias[o]) : 0.0;
                    for (int f = 0; f < in_features; ++f) {
                        acc += aggregated[f] * static_cast<double>(wrow[f]);
                    }
                    orow[o] = static_cast<float>(acc);
                }
            }
        }
    }
    return 0;
}

// Sparse mixture of experts: top-k routing by softmax over the router, gate
// renormalization over the selected experts, and one expert forward pass per
// (token, selected expert). Unused experts are never evaluated.
// up_w is (experts, hidden, features), down_w is (experts, features, hidden).
int uzen_moe_f32(const float* x, const float* router, const float* up_w, const float* up_b,
                 const float* down_w, const float* down_b, float* out, int tokens, int features,
                 int experts, int hidden, int top_k, int activation) {
    if (!valid(x) || !valid(router) || !valid(up_w) || !valid(down_w) || !valid(out) ||
        tokens <= 0 || features <= 0 || experts <= 0 || hidden <= 0 || top_k < 1 ||
        top_k > experts) {
        return 1;
    }
#pragma omp parallel for schedule(static) if (tokens > 1)
    for (int token = 0; token < tokens; ++token) {
        const float* xrow = x + static_cast<std::size_t>(token) * features;
        std::vector<double> logits(static_cast<std::size_t>(experts), 0.0);
        double maximum = -std::numeric_limits<double>::infinity();
        for (int e = 0; e < experts; ++e) {
            const float* rrow = router + static_cast<std::size_t>(e) * features;
            double acc = 0.0;
            for (int f = 0; f < features; ++f) {
                acc += static_cast<double>(xrow[f]) * static_cast<double>(rrow[f]);
            }
            logits[e] = acc;
            maximum = std::max(maximum, acc);
        }
        double total = 0.0;
        for (int e = 0; e < experts; ++e) {
            logits[e] = std::exp(logits[e] - maximum);
            total += logits[e];
        }
        if (total > 0.0) {
            for (int e = 0; e < experts; ++e) {
                logits[e] /= total;
            }
        }
        // Deterministic top-k: repeatedly take the largest remaining probability
        // (ties resolve to the lowest expert index).
        std::vector<int> selected(static_cast<std::size_t>(top_k), -1);
        std::vector<char> taken(static_cast<std::size_t>(experts), 0);
        for (int slot = 0; slot < top_k; ++slot) {
            int best = -1;
            double best_value = -std::numeric_limits<double>::infinity();
            for (int e = 0; e < experts; ++e) {
                if (!taken[e] && logits[e] > best_value) {
                    best_value = logits[e];
                    best = e;
                }
            }
            selected[slot] = best;
            if (best >= 0) {
                taken[best] = 1;
            }
        }
        double gate_total = 0.0;
        for (int slot = 0; slot < top_k; ++slot) {
            if (selected[slot] >= 0) {
                gate_total += logits[selected[slot]];
            }
        }
        const double normalizer = gate_total > 0.0 ? gate_total : 1.0;
        float* orow = out + static_cast<std::size_t>(token) * features;
        std::vector<double> accumulator(static_cast<std::size_t>(features), 0.0);
        std::vector<double> hidden_state(static_cast<std::size_t>(hidden), 0.0);
        for (int slot = 0; slot < top_k; ++slot) {
            const int expert = selected[slot];
            if (expert < 0) {
                continue;
            }
            const double gate = logits[expert] / normalizer;
            const float* up = up_w + static_cast<std::size_t>(expert) * hidden * features;
            for (int h = 0; h < hidden; ++h) {
                const float* urow = up + static_cast<std::size_t>(h) * features;
                double acc = up_b ? static_cast<double>(
                                        up_b[static_cast<std::size_t>(expert) * hidden + h])
                                  : 0.0;
                for (int f = 0; f < features; ++f) {
                    acc += static_cast<double>(xrow[f]) * static_cast<double>(urow[f]);
                }
                hidden_state[h] = activate(acc, activation);
            }
            const float* down =
                down_w + static_cast<std::size_t>(expert) * features * hidden;
            for (int f = 0; f < features; ++f) {
                const float* drow = down + static_cast<std::size_t>(f) * hidden;
                double acc = down_b ? static_cast<double>(
                                          down_b[static_cast<std::size_t>(expert) * features + f])
                                    : 0.0;
                for (int h = 0; h < hidden; ++h) {
                    acc += hidden_state[h] * static_cast<double>(drow[h]);
                }
                accumulator[f] += gate * acc;
            }
        }
        for (int f = 0; f < features; ++f) {
            orow[f] = static_cast<float>(accumulator[f]);
        }
    }
    return 0;
}

// ------------------------------------------------------------------------- //
// int8 quantized linear
// ------------------------------------------------------------------------- //

// y = x @ (w_q * scale)^T + b with int8 weights and per-output-channel scales.
// The dequantization happens inside the accumulation, never by materializing a
// float weight matrix.
int uzen_quantized_linear_f32(const float* x, const std::int8_t* wq, const float* scale,
                              const float* bias, float* out, int rows, int in_features,
                              int out_features) {
    if (!valid(x) || !valid(wq) || !valid(scale) || !valid(out) || rows < 0 ||
        in_features <= 0 || out_features <= 0) {
        return 1;
    }
    if (rows == 0) {
        return 0;
    }
#pragma omp parallel for schedule(static) if (rows > 1)
    for (int row = 0; row < rows; ++row) {
        const float* xrow = x + static_cast<std::size_t>(row) * in_features;
        float* orow = out + static_cast<std::size_t>(row) * out_features;
        for (int o = 0; o < out_features; ++o) {
            const std::int8_t* wrow = wq + static_cast<std::size_t>(o) * in_features;
            double acc = 0.0;
            for (int k = 0; k < in_features; ++k) {
                acc += static_cast<double>(xrow[k]) * static_cast<double>(wrow[k]);
            }
            acc *= static_cast<double>(scale[o]);
            if (bias) {
                acc += static_cast<double>(bias[o]);
            }
            orow[o] = static_cast<float>(acc);
        }
    }
    return 0;
}

// Reduction over an arbitrary axis, expressed as (outer, axis, inner) strides.
// This covers pooling over the sequence axis, which the last-axis reduction in
// kernels.cpp cannot express. mode 0 = mean, mode 1 = sum.
int uzen_reduce_axis_f32(const float* x, float* out, int outer, int axis, int inner,
                         int mode) {
    if (!valid(x) || !valid(out) || outer < 0 || axis <= 0 || inner < 0 ||
        (mode != 0 && mode != 1)) {
        return 1;
    }
    if (outer == 0 || inner == 0) {
        return 0;
    }
#pragma omp parallel for schedule(static) if (outer > 1)
    for (int o = 0; o < outer; ++o) {
        const float* base = x + static_cast<std::size_t>(o) * axis * inner;
        float* obase = out + static_cast<std::size_t>(o) * inner;
        for (int i = 0; i < inner; ++i) {
            double total = 0.0;
            const float* cursor = base + i;
            for (int a = 0; a < axis; ++a) {
                total += static_cast<double>(cursor[static_cast<std::size_t>(a) * inner]);
            }
            obase[i] = static_cast<float>(mode == 0 ? total / axis : total);
        }
    }
    return 0;
}

// ------------------------------------------------------------------------- //
// single-pass attention
// ------------------------------------------------------------------------- //

// Flash-style attention: one pass over key blocks with an online softmax, so the
// (q_seq, kv_seq) score matrix is never materialized and each score is computed
// once. Causal, sliding-window and ALiBi masks match uzen_attention_f32.
int uzen_attention_flash_f32(const float* q, const float* k, const float* v, float* out,
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
    constexpr int kKeyBlock = 64;
#pragma omp parallel for collapse(2) schedule(static) if (batch * heads > 1)
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            std::vector<double> accumulator(static_cast<std::size_t>(dimensions), 0.0);
            std::vector<double> block_scores(static_cast<std::size_t>(kKeyBlock), 0.0);
            for (int t = 0; t < q_seq; ++t) {
                const std::size_t qbase =
                    ((static_cast<std::size_t>(b) * heads + h) * q_seq + t) * dimensions;
                const float* qrow = q + qbase;
                const int qpos = q_offset + t;
                std::fill(accumulator.begin(), accumulator.end(), 0.0);
                double running_max = -std::numeric_limits<double>::infinity();
                double running_sum = 0.0;
                for (int start = 0; start < kv_seq; start += kKeyBlock) {
                    const int stop = std::min(start + kKeyBlock, kv_seq);
                    double block_max = -std::numeric_limits<double>::infinity();
                    for (int s = start; s < stop; ++s) {
                        if (causal && s > qpos) {
                            block_scores[s - start] = -std::numeric_limits<double>::infinity();
                            continue;
                        }
                        if (window >= 0 && std::abs(qpos - s) > window) {
                            block_scores[s - start] = -std::numeric_limits<double>::infinity();
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
                        block_scores[s - start] = score;
                        block_max = std::max(block_max, score);
                    }
                    if (!std::isfinite(block_max)) {
                        continue;
                    }
                    const double new_max = std::isfinite(running_max)
                                               ? std::max(running_max, block_max)
                                               : block_max;
                    const double rescale = std::isfinite(running_max)
                                               ? std::exp(running_max - new_max)
                                               : 0.0;
                    if (rescale != 1.0) {
                        for (int d = 0; d < dimensions; ++d) {
                            accumulator[d] *= rescale;
                        }
                        running_sum *= rescale;
                    }
                    running_max = new_max;
                    for (int s = start; s < stop; ++s) {
                        const double score = block_scores[s - start];
                        if (!std::isfinite(score)) {
                            continue;
                        }
                        const double weight = std::exp(score - running_max);
                        running_sum += weight;
                        const std::size_t vbase =
                            ((static_cast<std::size_t>(b) * heads + h) * kv_seq + s) * dimensions;
                        const float* vrow = v + vbase;
                        for (int d = 0; d < dimensions; ++d) {
                            accumulator[d] += weight * static_cast<double>(vrow[d]);
                        }
                    }
                }
                float* orow = out + qbase;
                if (running_sum <= 0.0) {
                    for (int d = 0; d < dimensions; ++d) {
                        orow[d] = 0.0f;
                    }
                } else {
                    for (int d = 0; d < dimensions; ++d) {
                        orow[d] = static_cast<float>(accumulator[d] / running_sum);
                    }
                }
            }
        }
    }
    return 0;
}

}  // extern "C"
