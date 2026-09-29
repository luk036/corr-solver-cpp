/** @file sites.hpp
 *  @brief Site layouts and biased sample covariance generation.
 */

#pragma once

#include <cstddef>
#include <ellalgo/arr.hpp>
#include <random>

/// Uniform 2D grid of nx by ny sites over [0,10] x [0,8].
Arr create_2d_sites(size_t nx = 10U, size_t ny = 8U);

/// Biased sample covariance for an exponential kernel over site distances.
/// When @p rng is null a fresh engine seeded with 5 is used, which keeps the
/// historical deterministic output.
Arr create_2d_isotropic(const Arr& site, size_t N = 3000U, std::mt19937_64* rng = nullptr);

/// Average @p N draws of @f$ y y^T @f$ with @f$ y \sim N(0, var^2 \Sigma + tau^2 I) @f$.
/// When @p rng is null a fresh engine seeded with 5 is used, keeping the
/// deterministic output.
Arr sample_covariance(const Arr& Sigma, size_t N, std::mt19937_64* rng = nullptr,
                      double var = 2.0, double tau = 0.00001);
