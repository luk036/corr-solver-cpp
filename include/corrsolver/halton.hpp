/** @file halton.hpp
 *  @brief Self-contained Halton low-discrepancy point generator.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <ellalgo/arr.hpp>

/// Radical inverse of i in the given base.
inline double radical_inverse(uint64_t i, uint64_t base) {
    double f = 1.0 / static_cast<double>(base);
    double r = 0.0;
    while (i > 0) {
        r += f * static_cast<double>(i % base);
        i /= base;
        f /= static_cast<double>(base);
    }
    return r;
}

/// Halton sites scaled by (10, 8); point k (k = 1..nx*ny) is
/// (radical_inverse(k, 2) * 10, radical_inverse(k, 3) * 8).
inline Arr create_2d_sites_halton(size_t nx, size_t ny) {
    const size_t num_grid = nx * ny;
    constexpr double s_end[2] = {10.0, 8.0};
    constexpr uint64_t bases[2] = {2U, 3U};
    Arr site(num_grid, 2);
    for (size_t k = 1; k <= num_grid; ++k) {
        for (size_t d = 0; d < 2; ++d)
            site(k - 1, d) = radical_inverse(static_cast<uint64_t>(k), bases[d]) * s_end[d];
    }
    return site;
}
