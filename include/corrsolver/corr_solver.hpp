/** @file corr_solver.hpp
 *  @brief Fitting correlation/error-correction polynomials to biased covariance matrices.
 */

#pragma once

#include <cstddef>
#include <ellalgo/arr.hpp>
#include <optional>
#include <vector>

/// Outcome of a correlation fit. `ok` is false when the cutting-plane search
/// failed, in which case `coeffs` is empty or stale.
struct FitResult {
    Arr coeffs;
    size_t iters = 0;
    bool ok = false;
};

/// Polynomial basis: Sigma_k = D.^k, the Hadamard powers of the distance matrix.
std::vector<Arr> construct_poly_matrix(const Arr& site, size_t m);

/// Assemble Omega(x) = sum_i x_i Sigma_i.
Arr corr_omega(const Arr& x, const std::vector<Arr>& Sig);

/// MLE objective log det Omega(x) + Tr(Omega(x)^-1 Y).
double corr_mle_obj(const Arr& x, const std::vector<Arr>& Sig, const Arr& Y);

/// Evaluate the polynomial with ascending coefficients c at every point of x.
Arr eval_poly_curve(const Arr& c, const Arr& x);

/// Least-squares fit with the augmented (coeffs..., t) variable.
FitResult lsq_corr_poly2(const Arr& Y, const Arr& site, size_t m);

/// Least-squares fit on an explicit basis; enforces monotone coefficients on
/// the leading entries when @p n_coeff is set.
FitResult lsq_corr_generic(const Arr& Y, const std::vector<Arr>& Sigma,
                           std::optional<size_t> n_coeff);

/// Least-squares fit on the quadratic B-spline basis.
FitResult lsq_corr_bspline(const Arr& Y, const Arr& site, size_t m);

/// Maximum-likelihood fit subject to 2Y >= Omega >= 0.
FitResult mle_corr_poly(const Arr& Y, const Arr& site, size_t m);

/// CCP fit with the polynomial basis, warm-started from the least-squares fit.
FitResult cccp_corr_poly(const Arr& Y, const Arr& site, size_t m);

/// One CCP round on an explicit basis.
FitResult cccp_corr_step(const std::vector<Arr>& Sig, const Arr& Y, Arr x,
                         std::optional<size_t> n_coeff);

/// CCP fit on an explicit basis until the objective stalls.
FitResult cccp_corr_generic(const std::vector<Arr>& Sig, const Arr& Y, Arr x,
                            std::optional<size_t> n_coeff);

/// CCP fit on the quadratic B-spline basis.
FitResult cccp_corr_bspline(const Arr& Y, const Arr& site, size_t m);
