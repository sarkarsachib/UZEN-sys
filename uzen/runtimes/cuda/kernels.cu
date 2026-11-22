// UZEN CUDA kernels (float32).
//
// This file is the real device implementation behind the `cuda` runtime. It is
// deliberately self-contained: no includes, no C++ runtime, only the identifiers
// NVRTC/nvcc predefine for device code. That keeps the same source compilable by
// NVRTC at runtime (users without a CUDA toolkit) and by nvcc when one is present.
//
// Coverage is the float32 arithmetic the CPU native executor covers: linear,
// matmul, layernorm, rmsnorm, softmax, GELU (tanh/exact), SiLU, ReLU, tanh,
// sigmoid, same-shape add/sub/mul, scale, clip, embedding, last-axis mean/sum and
// masked attention. Layout operators are host-side byte movement by design.
//
// There are no tensor-core, half-precision or multi-GPU paths here. A kernel is
// only ever claimed when it exists in this file and was compiled.

extern "C" {

__device__ float uzen_block_sum(float value, float* scratch) {
    int tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    return scratch[0];
}

__device__ float uzen_block_max(float value, float* scratch) {
    int tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }
    return scratch[0];
}

// C[rows, out] = A[rows, in] * W[out, in]^T + bias
__global__ void uzen_linear_f32(
    const float* a, const float* w, const float* bias, float* c,
    int rows, int in_features, int out_features) {
    __shared__ float tile_a[16][16];
    __shared__ float tile_b[16][16];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * 16 + ty;
    int col = blockIdx.x * 16 + tx;
    float acc = 0.0f;
    for (int base = 0; base < in_features; base += 16) {
        tile_a[ty][tx] = (row < rows && base + tx < in_features)
            ? a[row * in_features + base + tx] : 0.0f;
        tile_b[ty][tx] = (col < out_features && base + ty < in_features)
            ? w[col * in_features + base + ty] : 0.0f;
        __syncthreads();
        for (int k = 0; k < 16; ++k) {
            acc += tile_a[ty][k] * tile_b[k][tx];
        }
        __syncthreads();
    }
    if (row < rows && col < out_features) {
        c[row * out_features + col] = acc + (bias ? bias[col] : 0.0f);
    }
}

// C[rows, cols] = A[rows, k] * B[k, cols]
__global__ void uzen_matmul_f32(
    const float* a, const float* b, float* c,
    int rows, int inner, int cols) {
    __shared__ float tile_a[16][16];
    __shared__ float tile_b[16][16];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * 16 + ty;
    int col = blockIdx.x * 16 + tx;
    float acc = 0.0f;
    for (int base = 0; base < inner; base += 16) {
        tile_a[ty][tx] = (row < rows && base + tx < inner)
            ? a[row * inner + base + tx] : 0.0f;
        tile_b[ty][tx] = (base + ty < inner && col < cols)
            ? b[(base + ty) * cols + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < 16; ++k) {
            acc += tile_a[ty][k] * tile_b[k][tx];
        }
        __syncthreads();
    }
    if (row < rows && col < cols) {
        c[row * cols + col] = acc;
    }
}

__global__ void uzen_layernorm_f32(
    const float* x, const float* weight, const float* bias, float* out,
    int rows, int cols, float eps) {
    __shared__ float scratch[256];
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* source = x + (long long)row * cols;
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float value = source[i];
        sum += value;
        sum_sq += value * value;
    }
    sum = uzen_block_sum(sum, scratch);
    __syncthreads();
    sum_sq = uzen_block_sum(sum_sq, scratch);
    float mean = sum / (float)cols;
    float variance = fmaxf(sum_sq / (float)cols - mean * mean, 0.0f);
    float inverse = rsqrtf(variance + eps);
    float* destination = out + (long long)row * cols;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float value = (source[i] - mean) * inverse;
        if (weight) value *= weight[i];
        if (bias) value += bias[i];
        destination[i] = value;
    }
}

__global__ void uzen_rmsnorm_f32(
    const float* x, const float* weight, float* out,
    int rows, int cols, float eps) {
    __shared__ float scratch[256];
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* source = x + (long long)row * cols;
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float value = source[i];
        sum_sq += value * value;
    }
    sum_sq = uzen_block_sum(sum_sq, scratch);
    float inverse = rsqrtf(sum_sq / (float)cols + eps);
    float* destination = out + (long long)row * cols;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float value = source[i] * inverse;
        if (weight) value *= weight[i];
        destination[i] = value;
    }
}

__global__ void uzen_softmax_f32(const float* x, float* out, int rows, int cols) {
    __shared__ float scratch[256];
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* source = x + (long long)row * cols;
    float local_max = -1e30f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        local_max = fmaxf(local_max, source[i]);
    }
    float maximum = uzen_block_max(local_max, scratch);
    __syncthreads();
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        local_sum += expf(source[i] - maximum);
    }
    float total = uzen_block_sum(local_sum, scratch);
    float* destination = out + (long long)row * cols;
    float inverse = total > 0.0f ? 1.0f / total : 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        destination[i] = expf(source[i] - maximum) * inverse;
    }
}

__global__ void uzen_gelu_tanh_f32(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float value = x[i];
    float inner = 0.7978845608028654f * (value + 0.044715f * value * value * value);
    out[i] = 0.5f * value * (1.0f + tanhf(inner));
}

__global__ void uzen_gelu_exact_f32(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float value = x[i];
    out[i] = 0.5f * value * (1.0f + erff(value * 0.7071067811865476f));
}

__global__ void uzen_silu_f32(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float value = x[i];
    out[i] = value / (1.0f + expf(-value));
}

__global__ void uzen_relu_f32(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = fmaxf(x[i], 0.0f);
}

__global__ void uzen_tanh_f32(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = tanhf(x[i]);
}

__global__ void uzen_sigmoid_f32(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = 1.0f / (1.0f + expf(-x[i]));
}

// op: 0 add, 1 subtract, 2 multiply. Shapes must match exactly.
__global__ void uzen_binary_f32(
    const float* a, const float* b, float* out, int n, int op) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (op == 0) out[i] = a[i] + b[i];
    else if (op == 1) out[i] = a[i] - b[i];
    else out[i] = a[i] * b[i];
}

__global__ void uzen_scale_f32(const float* x, float* out, int n, float factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = x[i] * factor;
}

__global__ void uzen_clip_f32(
    const float* x, float* out, int n, float minimum, float maximum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = fminf(fmaxf(x[i], minimum), maximum);
}

__global__ void uzen_embedding_f32(
    const long long* indices, const float* weight, float* out,
    int tokens, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tokens * dim) return;
    int token = i / dim;
    int column = i - token * dim;
    long long row = indices[token];
    out[i] = weight[row * dim + column];
}

// mode: 0 mean, 1 sum over the last axis.
__global__ void uzen_reduce_last_f32(
    const float* x, float* out, int rows, int cols, int mode) {
    __shared__ float scratch[256];
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* source = x + (long long)row * cols;
    float total = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        total += source[i];
    }
    total = uzen_block_sum(total, scratch);
    if (threadIdx.x == 0) {
        out[row] = mode == 0 ? total / (float)cols : total;
    }
}

// Attention for (batch, heads, q_seq, dim) against (batch, heads, kv_seq, dim).
// Causal masks use the KV-cache offset (kv_seq - q_seq) so a single-token decode
// can attend to every cached key up to and including itself. A non-negative
// window keeps keys within that absolute distance of the query, matching the CPU
// runtime. ALiBi subtracts slope * distance per head, using absolute distance
// when attention is non-causal.
__global__ void uzen_attention_f32(
    const float* q, const float* k, const float* v, float* out,
    int batch, int heads, int q_seq, int kv_seq, int dim,
    float scale, int causal, int window, const float* alibi) {
    extern __shared__ float scores[];
    int query = blockIdx.x;
    int head = blockIdx.y;
    int item = blockIdx.z;
    if (item >= batch || head >= heads || query >= q_seq) return;

    long long q_base = ((long long)(item * heads + head) * q_seq + query) * dim;
    long long kv_base = ((long long)(item * heads + head) * kv_seq) * dim;
    int qpos = query + (kv_seq - q_seq);

    for (int key = threadIdx.x; key < kv_seq; key += blockDim.x) {
        const float* key_row = k + kv_base + (long long)key * dim;
        float dot = 0.0f;
        for (int d = 0; d < dim; ++d) {
            dot += q[q_base + d] * key_row[d];
        }
        float score = dot * scale;
        if (causal && key > qpos) score = -1e30f;
        if (window >= 0 && abs(qpos - key) > window) score = -1e30f;
        if (alibi) {
            int distance = causal ? (qpos - key) : abs(qpos - key);
            score -= alibi[head] * (float)distance;
        }
        scores[key] = score;
    }
    __syncthreads();

    float local_max = -1e30f;
    for (int key = threadIdx.x; key < kv_seq; key += blockDim.x) {
        local_max = fmaxf(local_max, scores[key]);
    }
    __shared__ float reduce_scratch[256];
    float maximum = uzen_block_max(local_max, reduce_scratch);
    // A fully masked row must produce zeros, like the CPU runtime, not a
    // uniform distribution over masked keys.
    if (maximum < -1e29f) maximum = 0.0f;
    __syncthreads();

    float local_sum = 0.0f;
    for (int key = threadIdx.x; key < kv_seq; key += blockDim.x) {
        float value = expf(scores[key] - maximum);
        scores[key] = value;
        local_sum += value;
    }
    float total = uzen_block_sum(local_sum, reduce_scratch);
    float inverse = total > 0.0f ? 1.0f / total : 0.0f;
    __syncthreads();
    for (int key = threadIdx.x; key < kv_seq; key += blockDim.x) {
        scores[key] *= inverse;
    }
    __syncthreads();

    long long out_base = q_base;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int key = 0; key < kv_seq; ++key) {
            acc += scores[key] * v[kv_base + (long long)key * dim + d];
        }
        out[out_base + d] = acc;
    }
}

// Inference batch normalization with stored running statistics. One block per row;
// mean/var are per-channel, so unlike layernorm this kernel never reduces over the
// row: it applies the frozen affine transform (x - mean) / sqrt(var + eps) * w + b.
__global__ void uzen_batchnorm_f32(
    const float* x, const float* weight, const float* bias,
    const float* running_mean, const float* running_var, float* out,
    int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* source = x + (long long)row * cols;
    float* destination = out + (long long)row * cols;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float value = (source[i] - running_mean[i]) * rsqrtf(running_var[i] + eps);
        if (weight) value *= weight[i];
        if (bias) value += bias[i];
        destination[i] = value;
    }
}

}  // extern "C"
