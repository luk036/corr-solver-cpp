#include <cmath>
#include <corrsolver/geometry.hpp>
#include <cstddef>

Arr construct_distance_matrix(const Arr& site) {
    auto n = site.rows();
    Arr D1(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double d = 0.0;
            for (size_t k = 0; k < site.cols(); ++k) {
                auto diff = site(j, k) - site(i, k);
                d += diff * diff;
            }
            D1(i, j) = std::sqrt(d);
            D1(j, i) = D1(i, j);
        }
    }
    return D1;
}
