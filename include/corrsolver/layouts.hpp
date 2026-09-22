/** @file layouts.hpp
 *  @brief Initial-guess strategies for the cutting-plane drivers.
 */

#pragma once

#include <cstddef>
#include <ellalgo/arr.hpp>
#include <ellalgo/ell.hpp>
#include <valarray>

/// Best-so-far objective seed for a fresh cutting-plane run.
constexpr double kInitialT = 1e100;

/// Initial ellipsoid: a center point plus either per-coordinate radii or a
/// scaling factor. An empty `radii` means the scaling factor is used.
struct InitialGuess {
    Arr x;
    std::valarray<double> radii;
    double alpha = 0.0;
};

/// Augmented (coeffs..., t) layout for the least-squares optimization.
InitialGuess lsq_initial_guess(const Arr& Y, size_t m);

/// Plain coefficient layout for the maximum-likelihood fit.
InitialGuess mle_initial_guess(size_t m);

/// Plain coefficient layout for one CCP round, centred on x.
InitialGuess cccp_initial_guess(const Arr& x);

/// Build the ellipsoid for a guess, consuming its center point.
Ell<Arr> make_ellipsoid(InitialGuess& guess);
