#pragma once

/// Minimal linear algebra helpers for Arr, replacing xtensor-blas operations.
/// For small FIR/correlation problems, naive O(n^3) is sufficient.

#include <ellalgo/arr.hpp>
#include <random>

// ---------------------------------------------------------------------------
// Matrix helpers
// ---------------------------------------------------------------------------

inline Arr transpose(const Arr& a) {
    assert(a.is_2d());
    Arr out(a.cols(), a.rows());
    for (size_t i = 0; i < a.rows(); ++i)
        for (size_t j = 0; j < a.cols(); ++j) out(j, i) = a(i, j);
    return out;
}

inline Arr flatten(const Arr& a) {
    Arr out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out(i) = a(i);
    return out;
}

inline Arr diagonal(const Arr& a) {
    assert(a.is_2d() && a.rows() == a.cols());
    auto n = a.rows();
    Arr out(n);
    for (size_t i = 0; i < n; ++i) out(i) = a(i, i);
    return out;
}

inline double trace(const Arr& a) { return sum(diagonal(a)); }

inline double norm(const Arr& a) { return std::sqrt(sum(a * a)); }

// ---------------------------------------------------------------------------
// Matrix-matrix multiplication: A * B  (A: m×k, B: k×n → result: m×n)
// ---------------------------------------------------------------------------
inline Arr matmul(const Arr& A, const Arr& B) {
    assert(A.is_2d() && B.is_2d() && A.cols() == B.rows());
    auto m = A.rows();
    auto k = A.cols();
    auto n = B.cols();
    Arr out(m, n);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (size_t t = 0; t < k; ++t) s += A(i, t) * B(t, j);
            out(i, j) = s;
        }
    return out;
}

// ---------------------------------------------------------------------------
// Cholesky decomposition: A = L * L^T  (A symmetric positive definite)
// Returns lower-triangular L.
// ---------------------------------------------------------------------------
inline Arr cholesky(const Arr& A) {
    auto n = A.rows();
    assert(A.is_2d() && A.cols() == n);
    Arr L(n, n);
    for (size_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (size_t k = 0; k < j; ++k) s += L(j, k) * L(j, k);
        L(j, j) = std::sqrt(A(j, j) - s);
        for (size_t i = j + 1; i < n; ++i) {
            s = 0.0;
            for (size_t k = 0; k < j; ++k) s += L(i, k) * L(j, k);
            L(i, j) = (A(i, j) - s) / L(j, j);
        }
    }
    return L;
}

// ---------------------------------------------------------------------------
// Matrix inverse via Cholesky (for SPD matrices).
// ---------------------------------------------------------------------------
inline Arr inv(const Arr& A) {
    auto n = A.rows();
    auto L = cholesky(A);
    // Solve L * Y = I  (forward substitution)
    Arr Y(n, n);
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < n; ++i) {
            double s = (i == j) ? 1.0 : 0.0;
            for (size_t k = 0; k < i; ++k) s -= L(i, k) * Y(k, j);
            Y(i, j) = s / L(i, i);
        }
    }
    // Solve L^T * X = Y  (back substitution)
    Arr X(n, n);
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = n; i-- > 0;) {
            double s = Y(i, j);
            for (size_t k = i + 1; k < n; ++k) s -= L(k, i) * X(k, j);
            X(i, j) = s / L(i, i);
        }
    }
    return X;
}

// ---------------------------------------------------------------------------
// Random number generation
// ---------------------------------------------------------------------------
inline std::mt19937_64& global_rng() {
    static std::mt19937_64 rng(std::random_device{}());
    return rng;
}

inline void random_seed(unsigned seed) { global_rng().seed(seed); }

inline Arr randn(size_t n) {
    auto& rng = global_rng();
    std::normal_distribution<double> dist(0.0, 1.0);
    Arr out(n);
    for (size_t i = 0; i < n; ++i) out(i) = dist(rng);
    return out;
}

// ---------------------------------------------------------------------------
// meshgrid: returns {XX, YY} where XX and YY are 2D grids
// ---------------------------------------------------------------------------
inline std::pair<Arr, Arr> meshgrid(const Arr& x, const Arr& y) {
    auto nx = x.size();
    auto ny = y.size();
    Arr xx(ny, nx);
    Arr yy(ny, nx);
    for (size_t i = 0; i < ny; ++i) {
        for (size_t j = 0; j < nx; ++j) {
            xx(i, j) = x(j);
            yy(i, j) = y(i);
        }
    }
    return {xx, yy};
}

// ---------------------------------------------------------------------------
// stack: combines multiple 1D arrays as rows (axis=0) into a 2D matrix
// ---------------------------------------------------------------------------
inline Arr stack(const Arr& a, const Arr& b, int /*unused*/ = 0) {
    assert(!a.is_2d() && !b.is_2d() && a.size() == b.size());
    Arr out(2, a.size());
    for (size_t j = 0; j < a.size(); ++j) {
        out(0, j) = a(j);
        out(1, j) = b(j);
    }
    return out;
}
