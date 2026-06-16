/** @file linalg.hpp
 *  @brief Minimal linear algebra helpers for Arr (transpose, matmul, Cholesky, inverse, etc.).
 */

#pragma once

/// Minimal linear algebra helpers for Arr, replacing xtensor-blas operations.
/// For small FIR/correlation problems, naive O(n^3) is sufficient.

#include <ellalgo/arr.hpp>
#include <random>

// ---------------------------------------------------------------------------
// Matrix helpers
// ---------------------------------------------------------------------------

/**
 * @brief Transpose a 2D matrix.
 * @param a Input matrix
 * @return Transposed matrix
 */
inline Arr transpose(const Arr& a) {
    assert(a.is_2d());
    Arr out(a.cols(), a.rows());
    for (size_t i = 0; i < a.rows(); ++i)
        for (size_t j = 0; j < a.cols(); ++j) out(j, i) = a(i, j);
    return out;
}

/**
 * @brief Flatten a matrix or vector into a 1D array.
 * @param a Input array
 * @return Flattened 1D array
 */
inline Arr flatten(const Arr& a) {
    Arr out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out(i) = a(i);
    return out;
}

/**
 * @brief Extract the diagonal of a square matrix.
 * @param a Input square matrix
 * @return 1D array containing diagonal elements
 */
inline Arr diagonal(const Arr& a) {
    assert(a.is_2d() && a.rows() == a.cols());
    auto n = a.rows();
    Arr out(n);
    for (size_t i = 0; i < n; ++i) out(i) = a(i, i);
    return out;
}

/**
 * @brief Compute the trace of a square matrix (sum of diagonal elements).
 * @param a Input square matrix
 * @return Trace value
 */
inline double trace(const Arr& a) { return sum(diagonal(a)); }

/**
 * @brief Compute the Frobenius norm of an array.
 * @param a Input array
 * @return Frobenius norm
 */
inline double norm(const Arr& a) { return std::sqrt(sum(a * a)); }

// ---------------------------------------------------------------------------
// Matrix-matrix multiplication: A * B  (A: m×k, B: k×n → result: m×n)
// ---------------------------------------------------------------------------

/**
 * @brief Multiply two matrices: A * B.
 * @param A Left matrix (m x k)
 * @param B Right matrix (k x n)
 * @return Result matrix (m x n)
 */
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

/**
 * @brief Cholesky decomposition: A = L * L^T for SPD matrices.
 * @param A Symmetric positive definite matrix
 * @return Lower-triangular Cholesky factor L
 */
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

/**
 * @brief Compute the inverse of a symmetric positive definite matrix via Cholesky.
 * @param A SPD matrix
 * @return Inverse matrix
 */
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

/**
 * @brief Get the global Mersenne Twister random number generator.
 * @return Reference to the global RNG
 */
inline std::mt19937_64& global_rng() {
    static std::mt19937_64 rng(std::random_device{}());
    return rng;
}

/**
 * @brief Seed the global random number generator.
 * @param seed The seed value
 */
inline void random_seed(unsigned seed) { global_rng().seed(seed); }

/**
 * @brief Generate a vector of standard normal random numbers.
 * @param n Number of elements
 * @return 1D array of i.i.d. N(0,1) samples
 */
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

/**
 * @brief Create 2D meshgrid arrays from 1D coordinate vectors.
 * @param x 1D x-coordinates
 * @param y 1D y-coordinates
 * @return Pair of 2D arrays (XX, YY)
 */
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

/**
 * @brief Stack two 1D arrays as rows into a 2D matrix.
 * @param a First row
 * @param b Second row
 * @return 2D matrix with two rows
 */
inline Arr stack(const Arr& a, const Arr& b, int /*unused*/ = 0) {
    assert(!a.is_2d() && !b.is_2d() && a.size() == b.size());
    Arr out(2, a.size());
    for (size_t j = 0; j < a.size(); ++j) {
        out(0, j) = a(j);
        out(1, j) = b(j);
    }
    return out;
}
