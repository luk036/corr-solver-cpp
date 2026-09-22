/** @file types.hpp
 *  @brief Shared type aliases.
 */

#pragma once

#include <ellalgo/arr.hpp>
#include <utility>

/// A separating cut: (subgradient, violation).
using Cut = std::pair<Arr, double>;
