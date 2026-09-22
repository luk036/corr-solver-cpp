/** @file geometry.hpp
 *  @brief Pairwise distance geometry for site layouts.
 */

#pragma once

#include <ellalgo/arr.hpp>

/// Euclidean distance matrix: D(i,j) = ||site(i) - site(j)||.
Arr construct_distance_matrix(const Arr& site);
