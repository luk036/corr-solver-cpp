#include <corrsolver/corr_solver.hpp>
#include <corrsolver/bspline.hpp>
#include <corrsolver/layouts.hpp>
#include <corrsolver/linalg.hpp>
#include <corrsolver/oracles.hpp>
#include <cmath>
#include <cstddef>
#include <ellalgo/cutting_plane.hpp>
#include <ellalgo/ell.hpp>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

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

Arr corr_omega(const Arr& x, const std::vector<Arr>& Sig) {
    auto n = Sig[0].rows();
    Arr Om(n, n);
    for (size_t r = 0; r < n; ++r)
        for (size_t c = 0; c < n; ++c) {
            double s = 0.0;
            for (size_t i = 0; i < x.size(); ++i) s += x(i) * Sig[i](r, c);
            Om(r, c) = s;
        }
    return Om;
}

double corr_mle_obj(const Arr& x, const std::vector<Arr>& Sig, const Arr& Y) {
    auto Om = corr_omega(x, Sig);
    auto L = cholesky(Om);
    double logdet = 0.0;
    for (size_t i = 0; i < L.rows(); ++i) logdet += 2.0 * std::log(L(i, i));
    return logdet + trace(matmul(inv(Om), Y));
}

Arr eval_poly_curve(const Arr& c, const Arr& x) {
    Arr out(x.size());
    for (size_t j = 0; j < x.size(); ++j) {
        double v = 0.0;
        for (size_t i = c.size(); i-- > 0;) v = v * x(j) + c(i);
        out(j) = v;
    }
    return out;
}

namespace {

/// Run a core against the raw oracle, or against the monotone-decorated one.
template <class Oracle, class Core>
auto with_mono(Oracle& omega, std::optional<size_t> n_coeff, Core&& core) {
    if (n_coeff) {
        auto wrapped = MonoDecreasingOracle<Oracle>(omega, n_coeff);
        return core(wrapped);
    }
    return core(omega);
}

FitResult run_lsq_core(const Arr& Y, size_t m, auto& omega, const SolverConfig& cfg) {
    auto guess = lsq_initial_guess(Y, m, cfg);
    auto ellip = make_ellipsoid(guess);
    auto t = kInitialT;
    auto [x_best, num_iters] = cutting_plane_optim(omega, ellip, t, cfg.options());
    if (x_best.size() != m + 1) return {Arr{}, num_iters, false};
    Arr a(m);
    for (size_t i = 0; i < m; ++i) a(i) = x_best(i);
    return {std::move(a), num_iters, true};
}

FitResult run_mle_core(size_t m, MleOracle& omega, const SolverConfig& cfg) {
    auto guess = mle_initial_guess(m, cfg);
    auto ellip = make_ellipsoid(guess);
    auto t = kInitialT;
    auto [x_best, num_iters] = cutting_plane_optim(omega, ellip, t, cfg.options());
    const bool ok = (x_best.size() == m);
    return {std::move(x_best), num_iters, ok};
}

}

FitResult lsq_corr_poly2(const Arr& Y, const Arr& site, size_t m, const SolverConfig& cfg) {
    auto Sig = construct_poly_matrix(site, m);
    auto omega = LsqOracle(Y.rows(), Sig, Y);
    return run_lsq_core(Y, m, omega, cfg);
}

FitResult lsq_corr_generic(const Arr& Y, const std::vector<Arr>& Sigma,
                           std::optional<size_t> n_coeff, const SolverConfig& cfg) {
    auto m = Sigma.size();
    auto omega = LsqOracle(Y.rows(), Sigma, Y);
    return with_mono(omega, n_coeff, [&](auto& o) { return run_lsq_core(Y, m, o, cfg); });
}

FitResult lsq_corr_bspline(const Arr& Y, const Arr& site, size_t m, const SolverConfig& cfg) {
    return lsq_corr_generic(Y, generate_bspline_info(site, m).Sigma, m, cfg);
}

FitResult mle_corr_poly(const Arr& Y, const Arr& site, size_t m, const SolverConfig& cfg) {
    auto Sig = construct_poly_matrix(site, m);
    auto omega = MleOracle(Y.rows(), Sig, Y);
    return run_mle_core(m, omega, cfg);
}

FitResult cccp_corr_poly(const Arr& Y, const Arr& site, size_t m, const SolverConfig& cfg) {
    auto Sig = construct_poly_matrix(site, m);
    auto lsq = lsq_corr_poly2(Y, site, m, cfg);
    return cccp_corr_generic(Sig, Y, std::move(lsq.coeffs), std::nullopt, cfg);
}

FitResult cccp_corr_step(const std::vector<Arr>& Sig, const Arr& Y, Arr x,
                         std::optional<size_t> n_coeff, const SolverConfig& cfg) {
    auto M = inv(corr_omega(x, Sig));
    auto omega = CccpMleOracle(Y.rows(), Sig, Y, M);
    return with_mono(omega, n_coeff, [&](auto& o) {
        auto guess = cccp_initial_guess(x, cfg);
        auto ellip = make_ellipsoid(guess);
        auto t = kInitialT;
        auto [x_new, iters] = cutting_plane_optim(o, ellip, t, cfg.options());
        const bool ok = (x_new.size() == x.size());
        return FitResult{std::move(x_new), iters, ok};
    });
}

FitResult cccp_corr_generic(const std::vector<Arr>& Sig, const Arr& Y, Arr x,
                            std::optional<size_t> n_coeff, const SolverConfig& cfg) {
    auto f_old = 1e100;
    size_t total_iters = 0;
    for (size_t k = 0; k < cfg.cccp_max_rounds; ++k) {
        auto step = cccp_corr_step(Sig, Y, x, n_coeff, cfg);
        total_iters += step.iters;
        if (!step.ok) break;
        auto f_new = corr_mle_obj(step.coeffs, Sig, Y);
        if (std::abs(f_old - f_new) < cfg.cccp_tol) {
            x = std::move(step.coeffs);
            break;
        }
        f_old = f_new;
        x = std::move(step.coeffs);
    }
    return {std::move(x), total_iters, true};
}

FitResult cccp_corr_bspline(const Arr& Y, const Arr& site, size_t m, const SolverConfig& cfg) {
    auto lsq = lsq_corr_bspline(Y, site, m, cfg);
    auto Sigma = generate_bspline_info(site, m).Sigma;
    return cccp_corr_generic(Sigma, Y, std::move(lsq.coeffs), m, cfg);
}
