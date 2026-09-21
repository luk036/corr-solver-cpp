/** @file bspline.hpp
 *  @brief Quadratic B-spline basis (clamped knots, scipy-compatible extrapolation)
 *         and the monotone-decreasing coefficient oracle.
 */

#pragma once

#include <cstddef>
#include <ellalgo/arr.hpp>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

using Cut = std::pair<Arr, double>;

struct BSplineInfo {
    std::vector<Arr> Sigma;
    Arr t;
    size_t k;
};

/// Defined in source/lsq_corr_ell.cpp.
Arr construct_distance_matrix(const Arr& site);

/// Clamped knot vector: (k+1) zeros, m-k-1 interior knots, (k+1) copies of dmax.
Arr clamped_knots(double dmax, size_t m, size_t k = 2);

/// Evaluate all m = t.size()-k-1 basis functions at every entry of D.
std::vector<Arr> eval_bspline_basis(const Arr& t, size_t k, const Arr& D);

/// Evaluate sum_i c(i) * B_i at every point of a 1-D grid x.
Arr eval_bspline_curve(const Arr& t, size_t k, const Arr& c, const Arr& x);

/// Build the clamped basis matrices and knot vector for a site set.
BSplineInfo generate_bspline_info(const Arr& site, size_t m);

/// Return the first monotonicity violation of x, or nullopt if non-increasing.
std::optional<Cut> mono_oracle(const Arr& x);

/// Enforce monotone non-increasing coefficients on the leading n_coeff entries,
/// then delegate to the wrapped basis oracle.
template <class Basis> class MonoDecreasingOracle2 {
    Basis& basis_;
    std::optional<size_t> n_coeff_;

  public:
    explicit MonoDecreasingOracle2(Basis& basis, std::optional<size_t> n_coeff = std::nullopt)
        : basis_(basis), n_coeff_(n_coeff) {}

    std::tuple<Cut, bool> assess_optim(const Arr& x, double& t) {
        auto n = x.size();
        auto k = n_coeff_.value_or(n > 0 ? n - 1 : 0);
        Arr xk(k);
        for (size_t i = 0; i < k; ++i) xk(i) = x(i);
        if (auto cut = mono_oracle(xk)) {
            Arr g = zeros(n);
            for (size_t i = 0; i < k; ++i) g(i) = cut->first(i);
            return {{std::move(g), cut->second}, false};
        }
        return basis_.assess_optim(x, t);
    }
};
