/** @file eigen.hpp
 *  @brief Symmetric Jacobi eigenvalue solver and design-matrix conditioning helper.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ellalgo/arr.hpp>
#include <vector>

/// Eigenvalues of a symmetric matrix, sorted ascending (cyclic Jacobi sweeps).
inline std::vector<double> jacobi_eigvals(const Arr& A) {
    auto n = A.rows();
    Arr a = A;
    for (size_t sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (size_t p = 0; p < n; ++p)
            for (size_t q = p + 1; q < n; ++q) off += a(p, q) * a(p, q);
        if (off <= 0.0) break;
        for (size_t p = 0; p < n; ++p) {
            for (size_t q = p + 1; q < n; ++q) {
                double apq = a(p, q);
                if (apq == 0.0) continue;
                double app = a(p, p);
                double aqq = a(q, q);
                double theta = (aqq - app) / (2.0 * apq);
                double sgn = (theta >= 0.0) ? 1.0 : -1.0;
                double tt = sgn / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                double c = 1.0 / std::sqrt(tt * tt + 1.0);
                double s = tt * c;
                for (size_t r = 0; r < n; ++r) {
                    if (r != p && r != q) {
                        double arp = a(r, p);
                        double arq = a(r, q);
                        a(r, p) = c * arp - s * arq;
                        a(p, r) = a(r, p);
                        a(r, q) = s * arp + c * arq;
                        a(q, r) = a(r, q);
                    }
                }
                a(p, p) = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                a(q, q) = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                a(p, q) = 0.0;
                a(q, p) = 0.0;
            }
        }
    }
    std::vector<double> ev(n);
    for (size_t i = 0; i < n; ++i) ev[i] = a(i, i);
    std::sort(ev.begin(), ev.end());
    return ev;
}

/// Smallest eigenvalue of a symmetric matrix.
inline double min_eig(const Arr& A) { return jacobi_eigvals(A).front(); }

namespace dd_detail {

struct DD {
    double hi = 0.0;
    double lo = 0.0;
};

inline DD quick_two_sum(double a, double b) {
    double s = a + b;
    return {s, b - (s - a)};
}

inline DD two_sum(double a, double b) {
    double s = a + b;
    double bb = s - a;
    return {s, (a - (s - bb)) + (b - bb)};
}

inline DD two_prod(double a, double b) {
    double p = a * b;
    return {p, std::fma(a, b, -p)};
}

inline DD dd_add(DD a, DD b) {
    DD s = two_sum(a.hi, b.hi);
    s.lo += a.lo + b.lo;
    return quick_two_sum(s.hi, s.lo);
}

inline DD dd_neg(DD a) { return {-a.hi, -a.lo}; }

inline DD dd_sub(DD a, DD b) { return dd_add(a, dd_neg(b)); }

inline DD dd_mul(DD a, DD b) {
    DD p = two_prod(a.hi, b.hi);
    p.lo += a.hi * b.lo + a.lo * b.hi;
    return quick_two_sum(p.hi, p.lo);
}

inline DD dd_div(DD a, DD b) {
    double q1 = a.hi / b.hi;
    DD r = dd_sub(a, dd_mul(b, {q1, 0.0}));
    double q2 = r.hi / b.hi;
    r = dd_sub(r, dd_mul(b, {q2, 0.0}));
    double q3 = r.hi / b.hi;
    DD q = two_sum(q1, q2);
    q.lo += q3;
    return quick_two_sum(q.hi, q.lo);
}

inline DD dd_sqrt(DD a) {
    if (a.hi <= 0.0) return {0.0, 0.0};
    double x = std::sqrt(a.hi);
    DD diff = dd_sub(a, two_prod(x, x));
    return quick_two_sum(x, diff.hi / (2.0 * x));
}

}

/// 2-norm condition number of the design matrix whose column k is vec(Sigma[k]).
/// The m x m Gram matrix and its Jacobi eigen-decomposition are carried in
/// double-double arithmetic: squaring the condition number pushes the smallest
/// eigenvalue below double precision, so plain double loses ~1e-5 relative.
inline double design_cond(const std::vector<Arr>& Sigma) {
    using dd_detail::DD;
    auto m = Sigma.size();
    auto n2 = Sigma[0].size();
    std::vector<DD> G(m * m);
    double dmax = 0.0;
    for (size_t p = 0; p < m; ++p) {
        for (size_t q = p; q < m; ++q) {
            DD s{0.0, 0.0};
            for (size_t i = 0; i < n2; ++i)
                s = dd_detail::dd_add(s, dd_detail::two_prod(Sigma[p](i), Sigma[q](i)));
            G[p * m + q] = s;
            G[q * m + p] = s;
            if (p == q) dmax = std::max(dmax, std::abs(s.hi));
        }
    }
    auto at = [&](size_t i, size_t j) -> DD& { return G[i * m + j]; };
    const double thresh = 1e-31 * dmax;
    for (size_t sweep = 0; sweep < 80; ++sweep) {
        bool changed = false;
        for (size_t p = 0; p < m; ++p) {
            for (size_t q = p + 1; q < m; ++q) {
                DD apq = at(p, q);
                if (std::abs(apq.hi) <= thresh) continue;
                changed = true;
                DD app = at(p, p);
                DD aqq = at(q, q);
                DD theta = dd_detail::dd_div(dd_detail::dd_sub(aqq, app), DD{2.0 * apq.hi, 2.0 * apq.lo});
                DD abtheta = (theta.hi >= 0.0) ? theta : dd_detail::dd_neg(theta);
                DD denom = dd_detail::dd_add(
                    abtheta,
                    dd_detail::dd_sqrt(dd_detail::dd_add(DD{1.0, 0.0}, dd_detail::dd_mul(theta, theta))));
                DD tt = dd_detail::dd_div(DD{(theta.hi >= 0.0) ? 1.0 : -1.0, 0.0}, denom);
                DD cc = dd_detail::dd_div(
                    DD{1.0, 0.0},
                    dd_detail::dd_sqrt(dd_detail::dd_add(DD{1.0, 0.0}, dd_detail::dd_mul(tt, tt))));
                DD ss = dd_detail::dd_mul(tt, cc);
                for (size_t r = 0; r < m; ++r) {
                    if (r == p || r == q) continue;
                    DD arp = at(r, p);
                    DD arq = at(r, q);
                    DD np = dd_detail::dd_sub(dd_detail::dd_mul(cc, arp), dd_detail::dd_mul(ss, arq));
                    DD nq = dd_detail::dd_add(dd_detail::dd_mul(ss, arp), dd_detail::dd_mul(cc, arq));
                    at(r, p) = np;
                    at(p, r) = np;
                    at(r, q) = nq;
                    at(q, r) = nq;
                }
                DD cc2 = dd_detail::dd_mul(cc, cc);
                DD ss2 = dd_detail::dd_mul(ss, ss);
                DD cs2 = dd_detail::dd_mul(DD{2.0, 0.0}, dd_detail::dd_mul(ss, cc));
                at(p, p) = dd_detail::dd_add(
                    dd_detail::dd_sub(dd_detail::dd_mul(cc2, app), dd_detail::dd_mul(cs2, apq)),
                    dd_detail::dd_mul(ss2, aqq));
                at(q, q) = dd_detail::dd_add(
                    dd_detail::dd_add(dd_detail::dd_mul(ss2, app), dd_detail::dd_mul(cs2, apq)),
                    dd_detail::dd_mul(cc2, aqq));
                at(p, q) = DD{0.0, 0.0};
                at(q, p) = DD{0.0, 0.0};
            }
        }
        if (!changed) break;
    }
    DD lmin = at(0, 0);
    DD lmax = at(0, 0);
    for (size_t i = 1; i < m; ++i) {
        if (at(i, i).hi < lmin.hi) lmin = at(i, i);
        if (at(i, i).hi > lmax.hi) lmax = at(i, i);
    }
    DD cond = dd_detail::dd_sqrt(dd_detail::dd_div(lmax, lmin));
    return cond.hi + cond.lo;
}
