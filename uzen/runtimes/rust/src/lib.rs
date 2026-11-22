//! UZEN Rust kernels.
//!
//! A small, dependency-free C-ABI library for the inference operators that are
//! cheap to express in Rust and easy to compare against the NumPy reference:
//! ``linear``, ``layernorm``, ``rmsnorm``, ``softmax``, ``gelu`` (tanh
//! approximation), ``silu``, ``scale``, ``add`` and ``mul``.
//!
//! The functions are deliberately scalar and obvious: this is a correctness-first
//! reference backend, not a SIMD kernel. Every function is ``unsafe`` at the FFI
//! boundary because it receives raw pointers; the Python bridge validates shapes and
//! dtypes before calling, and the loops never index outside the declared dimensions.

const VERSION: u32 = 1;

#[no_mangle]
pub extern "C" fn uzen_rust_version() -> u32 {
    VERSION
}

/// `out[r, o] = sum_i x[r, i] * w[o, i] + b[o]` (bias optional).
///
/// # Safety
/// All pointers must be valid for the element counts implied by the dimensions.
#[no_mangle]
pub unsafe extern "C" fn uzen_rust_linear(
    x: *const f32,
    w: *const f32,
    b: *const f32,
    out: *mut f32,
    rows: usize,
    in_features: usize,
    out_features: usize,
) {
    for r in 0..rows {
        for o in 0..out_features {
            let mut acc = 0.0f32;
            for i in 0..in_features {
                acc += *x.add(r * in_features + i) * *w.add(o * in_features + i);
            }
            if !b.is_null() {
                acc += *b.add(o);
            }
            *out.add(r * out_features + o) = acc;
        }
    }
}

/// `out[r, c] = (x[r, c] - mean_r) / sqrt(var_r + eps) * w[c] + b[c]`.
///
/// # Safety
/// Pointers must be valid for `rows * cols` elements (`w`/`b` for `cols`).
#[no_mangle]
pub unsafe extern "C" fn uzen_rust_layernorm(
    x: *const f32,
    w: *const f32,
    b: *const f32,
    out: *mut f32,
    rows: usize,
    cols: usize,
    eps: f32,
) {
    for r in 0..rows {
        let base = r * cols;
        let mut sum = 0.0f32;
        for c in 0..cols {
            sum += *x.add(base + c);
        }
        let mean = sum / cols as f32;
        let mut variance = 0.0f32;
        for c in 0..cols {
            let d = *x.add(base + c) - mean;
            variance += d * d;
        }
        variance /= cols as f32;
        let inverse = 1.0f32 / (variance + eps).sqrt();
        for c in 0..cols {
            let mut value = (*x.add(base + c) - mean) * inverse;
            if !w.is_null() {
                value *= *w.add(c);
            }
            if !b.is_null() {
                value += *b.add(c);
            }
            *out.add(base + c) = value;
        }
    }
}

/// `out[r, c] = x[r, c] / sqrt(mean(x[r, :]^2) + eps) * w[c]`.
///
/// # Safety
/// Pointers must be valid for `rows * cols` elements (`w` for `cols`).
#[no_mangle]
pub unsafe extern "C" fn uzen_rust_rmsnorm(
    x: *const f32,
    w: *const f32,
    out: *mut f32,
    rows: usize,
    cols: usize,
    eps: f32,
) {
    for r in 0..rows {
        let base = r * cols;
        let mut sum_sq = 0.0f32;
        for c in 0..cols {
            let value = *x.add(base + c);
            sum_sq += value * value;
        }
        let inverse = 1.0f32 / (sum_sq / cols.max(1) as f32 + eps).sqrt();
        for c in 0..cols {
            let mut value = *x.add(base + c) * inverse;
            if !w.is_null() {
                value *= *w.add(c);
            }
            *out.add(base + c) = value;
        }
    }
}

/// Numerically stable softmax over each row's last axis.
///
/// # Safety
/// Pointers must be valid for `rows * cols` elements.
#[no_mangle]
pub unsafe extern "C" fn uzen_rust_softmax(
    x: *const f32,
    out: *mut f32,
    rows: usize,
    cols: usize,
) {
    for r in 0..rows {
        let base = r * cols;
        let mut maximum = f32::NEG_INFINITY;
        for c in 0..cols {
            let value = *x.add(base + c);
            if value > maximum {
                maximum = value;
            }
        }
        let mut total = 0.0f32;
        for c in 0..cols {
            let value = (*x.add(base + c) - maximum).exp();
            *out.add(base + c) = value;
            total += value;
        }
        if total > 0.0 {
            for c in 0..cols {
                *out.add(base + c) /= total;
            }
        }
    }
}

/// GELU, tanh approximation (matches NIR's `gelu`).
///
/// # Safety
/// `x` must be readable and `out` writable for `n` elements.
#[no_mangle]
pub unsafe extern "C" fn uzen_rust_gelu(x: *const f32, out: *mut f32, n: usize) {
    let k = 0.7978845608028654f32;
    for i in 0..n {
        let v = *x.add(i);
        let inner = k * (v + 0.044715f32 * v * v * v);
        *out.add(i) = 0.5f32 * v * (1.0f32 + inner.tanh());
    }
}

/// SiLU: `x * sigmoid(x)`.
///
/// # Safety
/// `x` must be readable and `out` writable for `n` elements.
#[no_mangle]
pub unsafe extern "C" fn uzen_rust_silu(x: *const f32, out: *mut f32, n: usize) {
    for i in 0..n {
        let v = *x.add(i);
        *out.add(i) = v / (1.0f32 + (-v).exp());
    }
}

/// ReLU: `max(x, 0)`.
///
/// # Safety
/// `x` must be readable and `out` writable for `n` elements.
#[no_mangle]
pub unsafe extern "C" fn uzen_rust_relu(x: *const f32, out: *mut f32, n: usize) {
    for i in 0..n {
        let v = *x.add(i);
        *out.add(i) = if v > 0.0 { v } else { 0.0 };
    }
}

/// Hyperbolic tangent.
///
/// # Safety
/// `x` must be readable and `out` writable for `n` elements.
#[no_mangle]
pub unsafe extern "C" fn uzen_rust_tanh(x: *const f32, out: *mut f32, n: usize) {
    for i in 0..n {
        *out.add(i) = (*x.add(i)).tanh();
    }
}

/// Logistic sigmoid `1 / (1 + exp(-x))`.
///
/// # Safety
/// `x` must be readable and `out` writable for `n` elements.
#[no_mangle]
pub unsafe extern "C" fn uzen_rust_sigmoid(x: *const f32, out: *mut f32, n: usize) {
    for i in 0..n {
        *out.add(i) = 1.0f32 / (1.0f32 + (-*x.add(i)).exp());
    }
}

/// `out = x * factor`.
///
/// # Safety
/// `x` must be readable and `out` writable for `n` elements.
#[no_mangle]
pub unsafe extern "C" fn uzen_rust_scale(x: *const f32, out: *mut f32, n: usize, factor: f32) {
    for i in 0..n {
        *out.add(i) = *x.add(i) * factor;
    }
}

/// `out = a + b` elementwise (identical shapes).
///
/// # Safety
/// All pointers must be valid for `n` elements.
#[no_mangle]
pub unsafe extern "C" fn uzen_rust_add(a: *const f32, b: *const f32, out: *mut f32, n: usize) {
    for i in 0..n {
        *out.add(i) = *a.add(i) + *b.add(i);
    }
}

/// `out = a * b` elementwise (identical shapes).
///
/// # Safety
/// All pointers must be valid for `n` elements.
#[no_mangle]
pub unsafe extern "C" fn uzen_rust_mul(a: *const f32, b: *const f32, out: *mut f32, n: usize) {
    for i in 0..n {
        *out.add(i) = *a.add(i) * *b.add(i);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_matches_hand_computation() {
        let x = [1.0f32, 2.0];
        let w = [1.0f32, 0.0, 0.0, 1.0];
        let b = [0.5f32, -0.5];
        let mut out = [0.0f32; 2];
        unsafe { uzen_rust_linear(x.as_ptr(), w.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 1, 2, 2) };
        assert_eq!(out, [1.5, 1.5]);
    }

    #[test]
    fn softmax_sums_to_one() {
        let x = [1.0f32, 2.0, 3.0];
        let mut out = [0.0f32; 3];
        unsafe { uzen_rust_softmax(x.as_ptr(), out.as_mut_ptr(), 1, 3) };
        let total: f32 = out.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
    }
}
