/** @file kernels.hpp
 *  @brief Radial covariance kernels shared by the generators and experiments.
 */

#pragma once

#include <cmath>

/// Gaussian kernel exp(-rate * r^2), with r the distance.
inline double gaussian_kernel(double r, double rate) { return std::exp(-rate * r * r); }

/// Matern 1/2 (exponential) kernel exp(-rate * r), with r the distance.
inline double exponential_kernel(double r, double rate) { return std::exp(-rate * r); }
