/** @file bspline.cpp
 *  @brief Implementation of the quadratic B-spline basis and monotone oracle.
 */

#include <algorithm>
#include <cmath>
#include <corrsolver/bspline.hpp>
#include <stdexcept>
#include <string>

namespace {

// de Boor's algorithm for the k+1 nonzero basis functions of span i at u.
// u may lie outside [t(i), t(i+1)) to obtain the polynomial extension.
void basis_funs(double u, size_t i, size_t k, const Arr& t, std::vector<double>& N) {
    std::vector<double> left(k + 1);
    std::vector<double> right(k + 1);
    N.assign(k + 1, 0.0);
    N[0] = 1.0;
    for (size_t j = 1; j <= k; ++j) {
        left[j] = u - t(i + 1 - j);
        right[j] = t(i + j) - u;
        double saved = 0.0;
        for (size_t r = 0; r < j; ++r) {
            double temp = N[r] / (right[r + 1] + left[j - r]);
            N[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        N[j] = saved;
    }
}

size_t find_span(const Arr& t, size_t k, size_t n, double x) {
    if (x >= t(n)) return n - 1;
    if (x < t(k)) return k;
    size_t i = k;
    while (i + 1 < n && x >= t(i + 1)) ++i;
    return i;
}

}

Arr clamped_knots(double dmax, size_t m, size_t k) {
    auto full = linspace(0.0, dmax, m - k + 1);
    Arr t(m + k + 1);
    size_t pos = 0;
    for (size_t i = 0; i <= k; ++i) t(pos++) = 0.0;
    for (size_t i = 1; i + 1 < full.size(); ++i) t(pos++) = full(i);
    for (size_t i = 0; i <= k; ++i) t(pos++) = dmax;
    return t;
}

std::vector<Arr> eval_bspline_basis(const Arr& t, size_t k, const Arr& D) {
    auto n = t.size() - k - 1;
    auto nr = D.rows();
    auto nc = D.cols();
    std::vector<Arr> out;
    out.reserve(n);
    for (size_t b = 0; b < n; ++b) out.emplace_back(nr, nc);
    std::vector<double> basis;
    for (size_t r = 0; r < nr; ++r) {
        for (size_t c = 0; c < nc; ++c) {
            double x = D(r, c);
            auto i = find_span(t, k, n, x);
            basis_funs(x, i, k, t, basis);
            auto start = i - k;
            for (size_t j = 0; j <= k; ++j) out[start + j](r, c) = basis[j];
        }
    }
    return out;
}

Arr eval_bspline_curve(const Arr& t, size_t k, const Arr& c, const Arr& x) {
    auto n = t.size() - k - 1;
    Arr out(x.size());
    std::vector<double> basis;
    for (size_t j = 0; j < x.size(); ++j) {
        auto i = find_span(t, k, n, x(j));
        basis_funs(x(j), i, k, t, basis);
        auto start = i - k;
        double s = 0.0;
        for (size_t b = 0; b <= k; ++b) s += c(start + b) * basis[b];
        out(j) = s;
    }
    return out;
}

BSplineInfo generate_bspline_info(const Arr& site, size_t m) {
    const size_t k = 2;
    if (m < k + 1)
        throw std::invalid_argument("quadratic B-spline needs m >= " + std::to_string(k + 1) +
                                    " control points, got " + std::to_string(m));
    auto D = construct_distance_matrix(site);
    double dmax = 0.0;
    for (size_t i = 0; i < D.size(); ++i) dmax = std::max(dmax, D(i));
    auto t = clamped_knots(dmax, m, k);
    auto Sigma = eval_bspline_basis(t, k, D);
    return {std::move(Sigma), std::move(t), k};
}

std::optional<Cut> mono_oracle(const Arr& x) {
    auto n = x.size();
    Arr g = zeros(n);
    for (size_t i = 0; i + 1 < n; ++i) {
        auto fj = x(i + 1) - x(i);
        if (fj > 0.0) {
            g(i) = -1.0;
            g(i + 1) = 1.0;
            return Cut{std::move(g), fj};
        }
    }
    return std::nullopt;
}
