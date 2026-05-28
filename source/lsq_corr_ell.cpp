#include <cmath>
#include <corrsolver/linalg.hpp>
#include <corrsolver/qmi_oracle.hpp>
#include <cstddef>
#include <ellalgo/cutting_plane.hpp>
#include <ellalgo/ell.hpp>
#include <ellalgo/oracles/ldlt_mgr.hpp>
#include <ellalgo/oracles/lmi0_oracle.hpp>
#include <ellalgo/oracles/lmi_oracle.hpp>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

Arr create_2d_sites(size_t nx = 10U, size_t ny = 8U) {
    auto sx = linspace(0.0, 10.0, nx);
    auto sy = linspace(0.0, 8.0, ny);
    auto [xx, yy] = meshgrid(sx, sy);
    auto st = stack(flatten(xx), flatten(yy));
    return transpose(st);
}

Arr create_2d_isotropic(const Arr& site, size_t N = 3000U) {
    auto n = site.rows();
    const auto sdkern = 0.3;
    const auto var = 2.0;
    const auto tau = 0.00001;
    random_seed(5);

    Arr Sig(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i; j < n; ++j) {
            double d = 0.0;
            for (size_t k = 0; k < site.cols(); ++k) {
                auto diff = site(j, k) - site(i, k);
                d += diff * diff;
            }
            auto g = -sdkern * std::sqrt(d);
            Sig(i, j) = std::exp(g);
            Sig(j, i) = Sig(i, j);
        }
    }

    auto A = cholesky(Sig);
    Arr Y(n, n);
    for (size_t k = 0; k < N; ++k) {
        auto x = var * randn(n);
        // y = A * x
        Arr y(n);
        for (size_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (size_t j = 0; j < n; ++j) s += A(i, j) * x(j);
            y(i) = s;
        }
        // Add noise
        auto noise = randn(n);
        for (size_t i = 0; i < n; ++i) y(i) += tau * noise(i);
        // Y += y * y^T
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) Y(i, j) += y(i) * y(j);
    }
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) Y(i, j) /= static_cast<double>(N);
    return Y;
}

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

std::vector<Arr> construct_poly_matrix(const Arr& site, size_t m) {
    auto n = site.rows();
    auto D1 = construct_distance_matrix(site);
    auto D = ones(n, n);
    std::vector<Arr> Sig;
    Sig.reserve(m);
    for (size_t i = 0; i < m; ++i) {
        if (i > 0) {
            // D = D .* D1 (element-wise)
            for (size_t r = 0; r < n; ++r)
                for (size_t c = 0; c < n; ++c) D(r, c) *= D1(r, c);
        }
        Sig.emplace_back(D);
    }
    return Sig;
}

// === LsqOracle ===
class LsqOracle {
    using Cut = std::pair<Arr, double>;
    QmiOracle<Arr> _qmi;
    Lmi0Oracle<Arr> _lmi0;

  public:
    LsqOracle(size_t m, const std::vector<Arr>& F, const Arr& F0) : _qmi(F, F0), _lmi0(m, F) {}

    std::tuple<Cut, bool> assess_optim(const Arr& x, double& t) {
        auto n = x.size();
        Arr g = zeros(n);

        // v = x[0..n-2]
        Arr v(n - 1);
        for (size_t i = 0; i < n - 1; ++i) v(i) = x(i);

        if (auto* cut0 = this->_lmi0.assess_feas(v)) {
            const auto& [g0, f0] = *cut0;
            for (size_t i = 0; i < n - 1; ++i) g(i) = g0(i);
            g(n - 1) = 0.0;
            return {{std::move(g), f0}, false};
        }
        this->_qmi.update(x(n - 1));

        if (auto cut1 = this->_qmi.assess_feas(v)) {
            const auto& [g1, f1] = *cut1;
            const auto& Q = this->_qmi._mq;
            const auto& [start, stop] = Q.pos;
            Arr wit_vec = zeros(this->_qmi._m);
            Q.set_witness_vec(wit_vec);

            double v2norm2 = 0.0;
            for (size_t i = start; i < stop; ++i) v2norm2 += wit_vec(i) * wit_vec(i);

            for (size_t i = 0; i < n - 1; ++i) g(i) = g1(i);
            g(n - 1) = -v2norm2;
            return {{std::move(g), f1}, false};
        }
        g(n - 1) = 1.0;
        if (auto f0 = x(n - 1) - t; f0 > 0) {
            return {{std::move(g), f0}, false};
        }
        t = x(n - 1);
        return {{std::move(g), 0.0}, true};
    }
};

auto lsq_corr_core2(const Arr& Y, size_t m, LsqOracle& omega) {
    auto normY = 100.0 * norm(Y);
    auto normY2 = 32.0 * normY * normY;
    std::valarray<double> val(256.0, m + 1);
    val[m] = normY2 * normY2;
    Arr x = zeros(m + 1);
    x[0] = 4;
    x[m] = normY2 / 2.0;
    auto ellip = Ell<Arr>(val, x);
    auto t = 1e100;
    auto [x_best, num_iters] = cutting_plane_optim(omega, ellip, t);
    Arr a(m);
    for (size_t i = 0; i < m; ++i) a(i) = x_best(i);
    return std::make_tuple(std::move(a), num_iters);
}

std::tuple<Arr, size_t> lsq_corr_poly2(const Arr& Y, const Arr& site, size_t m) {
    auto Sig = construct_poly_matrix(site, m);
    auto omega = LsqOracle(Y.rows(), Sig, Y);
    return lsq_corr_core2(Y, m, omega);
}

// === MleOracle ===
class MleOracle {
    using Cut = std::pair<Arr, double>;
    Arr Y_;
    std::vector<Arr> sig_;
    Lmi0Oracle<Arr> _lmi0;
    LmiOracle<Arr> _lmi;

  public:
    MleOracle(size_t m, const std::vector<Arr>& Sig, const Arr& Y)
        : Y_{Y}, sig_{Sig}, _lmi0(m, Sig), _lmi(m, Sig, 2.0 * Y) {}

    std::tuple<Cut, bool> assess_optim(const Arr& x, double& t) {
        if (auto* cut1 = this->_lmi.assess_feas(x)) return {*cut1, false};
        if (auto* cut0 = this->_lmi0.assess_feas(x)) return {*cut0, false};

        auto n = x.size();
        auto m = this->Y_.rows();
        auto dim = this->_lmi0._mq._n;

        Arr R(dim, dim);
        this->_lmi0._mq.sqrt(R);
        auto invR = inv(R);
        auto S = matmul(invR, transpose(invR));
        auto SY = matmul(S, this->Y_);

        auto diag = diagonal(R);
        double log_sum = 0.0;
        for (size_t i = 0; i < diag.size(); ++i) log_sum += std::log(diag(i));
        auto f1 = 2.0 * log_sum + trace(SY);
        auto f = f1 - t;
        auto shrunk = false;
        if (f < 0.0) {
            t = f1;
            f = 0.0;
            shrunk = true;
        }

        Arr g = zeros(n);
        for (size_t i = 0; i < n; ++i) {
            auto SFsi = matmul(S, this->sig_[i]);
            auto tr = trace(SFsi);
            for (size_t k = 0; k < m; ++k) {
                // dot(row(SFsi, k), column(SY, k))
                for (size_t j = 0; j < m; ++j) tr -= SFsi(k, j) * SY(j, k);
            }
            g(i) = tr;
        }
        return {{std::move(g), f}, shrunk};
    }
};

auto mle_corr_core(size_t m, MleOracle& omega) {
    Arr x = zeros(m);
    x[0] = 4.0;
    auto ellip = Ell<Arr>(500.0, x);
    auto t = 1e100;
    return cutting_plane_optim(omega, ellip, t);
}

std::tuple<Arr, size_t> mle_corr_poly(const Arr& Y, const Arr& site, size_t m) {
    auto Sig = construct_poly_matrix(site, m);
    auto omega = MleOracle(Y.rows(), Sig, Y);
    return mle_corr_core(m, omega);
}
