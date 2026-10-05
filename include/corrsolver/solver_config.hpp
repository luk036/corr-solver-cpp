/** @file solver_config.hpp
 *  @brief Numeric configuration for the cutting-plane solver cores.
 */

#pragma once

#include <cstddef>
#include <ellalgo/ell_config.hpp>

/**
 * @brief Numeric constants for the cutting-plane / bisection cores.
 *
 * The initial radii are kept close to the coefficient scale: an ellipsoid much
 * larger than the solution wastes iterations (the method needs
 * @f$ O(n^2 \log(R/r)) @f$), while one that is too small makes the subproblem
 * infeasible. The MLE radius is shared by the CCP outer loop, which starts from
 * the least-squares solution rather than the origin.
 */
struct SolverConfig {
    double mle_r0 = 4.0;               ///< Initial MLE / CCP ellipsoid scale.
    double lsq_aug_r0 = 16.0;          ///< Initial ellipsoid scale for the augmented LSQ core.
    double lsq_frob_scale = 1.0;       ///< Multiplier on @f$ \|Y\|_F^2 @f$ for the augmented bound.
    double lsq_norm_scale = 1.0;       ///< Extra multiplicative inflation of @f$ \|Y\|_F @f$.
    double tolerance = 1e-12;          ///< Convergence tolerance forwarded to ellalgo.
    std::size_t max_iters = 2000;      ///< Iteration cap forwarded to ellalgo.
    std::size_t cccp_max_rounds = 50;  ///< Maximum CCP outer linearization rounds.
    double cccp_tol = 1e-8;            ///< CCP objective-change tolerance for early stopping.

    /// Build the ellalgo Options for the cutting-plane drivers.
    [[nodiscard]] auto options() const -> Options { return Options(max_iters, tolerance); }
};
