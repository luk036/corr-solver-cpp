// -*- coding: utf-8 -*-
#include <algorithm>
#include <cmath>
#include <corrsolver/bspline.hpp>
#include <corrsolver/eigen.hpp>
#include <corrsolver/halton.hpp>
#include <corrsolver/linalg.hpp>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

extern std::vector<Arr> construct_poly_matrix(const Arr&, size_t);
extern std::tuple<Arr, size_t> lsq_corr_generic(const Arr&, const std::vector<Arr>&, std::optional<size_t>);
extern std::tuple<Arr, size_t> cccp_corr_step(const std::vector<Arr>&, const Arr&, Arr, std::optional<size_t>);
extern std::tuple<Arr, size_t> lsq_corr_bspline(const Arr&, const Arr&, size_t);
extern std::tuple<Arr, size_t> cccp_corr_bspline(const Arr&, const Arr&, size_t);
extern Arr corr_omega(const Arr&, const std::vector<Arr>&);
extern double corr_mle_obj(const Arr&, const std::vector<Arr>&, const Arr&);

namespace {

std::filesystem::path project_root() {
    auto p = std::filesystem::current_path();
    while (true) {
        if (std::filesystem::exists(p / "xmake.lua")) return p;
        auto parent = p.parent_path();
        if (parent == p) return std::filesystem::current_path();
        p = parent;
    }
}

const size_t N_SITE = 20;
const size_t N_GRID = 1200;
const size_t N_INC_END = 1198;

double max_of(const Arr& a) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, a(i));
    return m;
}

struct Data {
    Arr site;
    Arr D;
    double dmax = 0.0;
    double dspan = 0.0;
    Arr true_cov;
    Arr xg;
    Arr ftrue;
};

Data make_data() {
    Data d;
    d.site = create_2d_sites_halton(5, 4);
    d.D = construct_distance_matrix(d.site);
    d.dmax = max_of(d.D);
    double s = 0.0;
    for (size_t j = 0; j < d.site.cols(); ++j) {
        double diff = d.site(d.site.rows() - 1, j) - d.site(0, j);
        s += diff * diff;
    }
    d.dspan = std::sqrt(s);
    d.true_cov = Arr(N_SITE, N_SITE);
    for (size_t i = 0; i < N_SITE; ++i)
        for (size_t j = 0; j < N_SITE; ++j) {
            double dd = d.D(i, j);
            d.true_cov(i, j) = 4.0 * std::exp(-0.12 * dd * dd);
        }
    d.xg = linspace(0.0, d.dmax, N_GRID);
    d.ftrue = Arr(N_GRID);
    for (size_t j = 0; j < N_GRID; ++j) d.ftrue(j) = 4.0 * std::exp(-0.12 * d.xg(j) * d.xg(j));
    return d;
}

Arr make_Y(const Arr& D, size_t N) {
    const size_t n = D.rows();
    Arr S(n, n);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            double dd = D(i, j);
            S(i, j) = std::exp(-0.12 * dd * dd);
        }
    auto A = cholesky(S);
    random_seed(5);
    Arr Y(n, n);
    for (size_t k = 0; k < N; ++k) {
        auto x = randn(n);
        for (size_t i = 0; i < n; ++i) x(i) *= 2.0;
        auto y = dot(A, x);
        auto noise = randn(n);
        for (size_t i = 0; i < n; ++i) y(i) += 1e-5 * noise(i);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) Y(i, j) += y(i) * y(j);
    }
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) Y(i, j) /= static_cast<double>(N);
    return Y;
}

Arr poly_curve(const Arr& c, const Arr& xg) {
    Arr out(xg.size());
    for (size_t j = 0; j < xg.size(); ++j) {
        double v = 0.0;
        for (size_t i = c.size(); i-- > 0;) v = v * xg(j) + c(i);
        out(j) = v;
    }
    return out;
}

size_t count_increasing(const Arr& curve) {
    size_t cnt = 0;
    for (size_t j = 0; j < N_INC_END && j + 1 < curve.size(); ++j)
        if (curve(j + 1) - curve(j) > 1e-9) ++cnt;
    return cnt;
}

struct Variant {
    std::string name;
    std::vector<Arr> Sigma;
    Arr t;
    size_t k = 2;
    std::optional<size_t> n_coeff;
    bool is_bs = false;
};

Variant make_poly(const Data& dat, size_t m) {
    Variant v;
    v.name = "poly" + std::to_string(m);
    v.Sigma = construct_poly_matrix(dat.site, m);
    return v;
}

Variant make_bs(const Data& dat, size_t m) {
    auto info = generate_bspline_info(dat.site, m);
    Variant v;
    v.name = "bs" + std::to_string(m);
    v.Sigma = std::move(info.Sigma);
    v.t = std::move(info.t);
    v.k = info.k;
    v.n_coeff = m;
    v.is_bs = true;
    return v;
}

Variant make_bs_old(const Data& dat, size_t m) {
    Variant v;
    v.name = "bs" + std::to_string(m) + "old";
    v.t = linspace(0.0, 1.2 * dat.dspan, m + 3);
    v.k = 2;
    v.Sigma = eval_bspline_basis(v.t, 2, dat.D);
    v.n_coeff = m;
    v.is_bs = true;
    return v;
}

struct Fit {
    bool ok = false;
    size_t iters = 0;
    double rel_err = 0.0;
    double min_eig = 0.0;
    size_t n_inc = 0;
    Arr coeffs;
};

Fit run_lsq(const Data& dat, const Variant& v, const Arr& Y) {
    Fit f;
    auto [c, iters] = lsq_corr_generic(Y, v.Sigma, v.n_coeff);
    f.iters = iters;
    if (c.size() != v.Sigma.size()) return f;
    f.ok = true;
    f.coeffs = c;
    auto Om = corr_omega(c, v.Sigma);
    f.rel_err = norm(Om - dat.true_cov) / norm(dat.true_cov);
    f.min_eig = min_eig(Om);
    auto curve = v.is_bs ? eval_bspline_curve(v.t, v.k, c, dat.xg) : poly_curve(c, dat.xg);
    f.n_inc = count_increasing(curve);
    return f;
}

void experiment1(const Data& dat, std::ofstream& csv) {
    std::printf("=== Experiment 1: design condition numbers ===\n");
    std::printf("%-8s %4s %18s\n", "family", "m", "cond");
    const size_t poly_ms[] = {2, 4, 6, 8, 10};
    for (size_t m : poly_ms) {
        double c = design_cond(construct_poly_matrix(dat.site, m));
        std::printf("%-8s %4zu %18.10g\n", "poly", m, c);
        csv << "poly," << m << "," << c << "\n";
    }
    const size_t bs_ms[] = {4, 6, 8, 10};
    for (size_t m : bs_ms) {
        auto info = generate_bspline_info(dat.site, m);
        double c = design_cond(info.Sigma);
        std::printf("%-8s %4zu %18.10g\n", "clamped", m, c);
        csv << "clamped," << m << "," << c << "\n";
    }
    for (size_t m : bs_ms) {
        auto t = linspace(0.0, 1.2 * dat.dspan, m + 3);
        auto Sig = eval_bspline_basis(t, 2, dat.D);
        double c = design_cond(Sig);
        std::printf("%-8s %4zu %18.10g\n", "legacy", m, c);
        csv << "legacy," << m << "," << c << "\n";
    }
}

void experiment2(const Data& dat) {
    std::printf("=== Experiment 2: knot diagnostic ===\n");
    std::printf("dmax=%.15g d_span=%.15g 1.2*d_span=%.15g\n", dat.dmax, dat.dspan, 1.2 * dat.dspan);
    auto t = linspace(0.0, 1.2 * dat.dspan, 7);
    std::printf("legacy knots m=4:");
    for (size_t i = 0; i < t.size(); ++i) std::printf(" %.15g", t(i));
    std::printf("\nlegacy valid domain m=4: [%.15g, %.15g]\n", t(2), t(4));
    size_t above_all = 0;
    size_t above_upper = 0;
    const size_t n = dat.D.rows();
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            if (dat.D(i, j) > t(4)) ++above_all;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
            if (dat.D(i, j) > t(4)) ++above_upper;
    const size_t upper_total = n * (n - 1) / 2;
    std::printf("entries > t(4): all=%zu/%zu (%.4f) upper=%zu/%zu (%.4f)\n", above_all, dat.D.size(),
                static_cast<double>(above_all) / static_cast<double>(dat.D.size()), above_upper,
                upper_total, static_cast<double>(above_upper) / static_cast<double>(upper_total));
    auto tc = clamped_knots(dat.dmax, 4, 2);
    std::printf("clamped knots m=4:");
    for (size_t i = 0; i < tc.size(); ++i) std::printf(" %.15g", tc(i));
    std::printf("\n");
}

void experiment3(const Data& dat, const std::vector<Variant>& vs, const Arr& Y, std::ofstream& csv) {
    std::printf("=== Experiment 3: fits at N=3000 ===\n");
    std::printf("%-8s %6s %14s %14s %6s\n", "variant", "iters", "rel_err", "min_eig", "nInc");
    for (const auto& v : vs) {
        auto f = run_lsq(dat, v, Y);
        if (!f.ok) {
            std::printf("%-8s FAIL\n", v.name.c_str());
            csv << v.name << ",FAIL,,,,\n";
            continue;
        }
        std::printf("%-8s %6zu %14.6g %14.6g %6zu\n", v.name.c_str(), f.iters, f.rel_err, f.min_eig,
                    f.n_inc);
        csv << v.name << "," << f.iters << "," << f.rel_err << "," << f.min_eig << "," << f.n_inc
            << "\n";
        if (v.is_bs) {
            std::printf("  %s coeffs:", v.name.c_str());
            for (size_t i = 0; i < f.coeffs.size(); ++i) std::printf(" %.10g", f.coeffs(i));
            std::printf("\n");
        }
    }
}

void run_ccp(const Variant& v, const Arr& Y, size_t N, std::ofstream& csv, const char* method,
             bool show_min_eig) {
    auto [x0, lsq_iters] = lsq_corr_generic(Y, v.Sigma, v.n_coeff);
    (void)lsq_iters;
    if (x0.size() != v.Sigma.size()) {
        std::printf("%-10s FAIL\n", method);
        csv << N << "," << method << ",FAIL,,,\n";
        return;
    }
    double f0 = corr_mle_obj(x0, v.Sigma, Y);
    Arr x = x0;
    double f_old = 1e100;
    size_t rounds = 0;
    double f1 = f0;
    for (size_t k = 0; k < 50; ++k) {
        auto [xn, iters] = cccp_corr_step(v.Sigma, Y, x, v.n_coeff);
        (void)iters;
        if (xn.size() != x.size()) break;
        double f = corr_mle_obj(xn, v.Sigma, Y);
        ++rounds;
        x = xn;
        f1 = f;
        if (std::abs(f_old - f) < 1e-8) break;
        f_old = f;
    }
    double me = min_eig(corr_omega(x, v.Sigma));
    if (show_min_eig)
        std::printf("%-10s rounds=%2zu f0=%.6g -> f1=%.6g min_eig=%.6g\n", method, rounds, f0, f1, me);
    else
        std::printf("%-10s rounds=%2zu f0=%.6g -> f1=%.6g\n", method, rounds, f0, f1);
    csv << N << "," << method << "," << rounds << "," << f0 << "," << f1 << "," << me << "\n";
}

void experiment4(const Variant& vpoly4, const Variant& vbs4, const Arr& Y, std::ofstream& csv) {
    std::printf("=== Experiment 4: CCP at N=3000 ===\n");
    run_ccp(vpoly4, Y, 3000, csv, "ccp_poly4", false);
    run_ccp(vbs4, Y, 3000, csv, "ccp_bs4", true);
}

void experiment5(const Data& dat, const std::vector<Variant>& vs, const Variant& vpoly4,
                 const Variant& vbs4, std::ofstream& csv) {
    std::printf("=== Experiment 5: N-sweep ===\n");
    for (size_t N = 1; N <= 50; ++N) {
        auto Y = make_Y(dat.D, N);
        std::printf("--- N=%zu ---\n", N);
        for (const auto& v : vs) {
            auto f = run_lsq(dat, v, Y);
            if (!f.ok) {
                std::printf("%-8s FAIL\n", v.name.c_str());
                csv << N << "," << v.name << ",FAIL,,,\n";
                continue;
            }
            std::printf("%-8s %6zu %14.6g %14.6g %6zu\n", v.name.c_str(), f.iters, f.rel_err,
                        f.min_eig, f.n_inc);
            csv << N << "," << v.name << "," << f.iters << "," << f.rel_err << "," << f.min_eig
                << "," << f.n_inc << "\n";
        }
        if (N == 1 || N == 5 || N == 10 || N == 20 || N == 50) {
            run_ccp(vpoly4, Y, N, csv, "ccp_poly4", false);
            run_ccp(vbs4, Y, N, csv, "ccp_bs4", true);
        }
    }
}

}

int main() {
    const Data dat = make_data();
    const auto vpoly4 = make_poly(dat, 4);
    const auto vpoly6 = make_poly(dat, 6);
    const auto vbs4 = make_bs(dat, 4);
    const auto vbs6 = make_bs(dat, 6);
    const auto vbs4old = make_bs_old(dat, 4);
    const std::vector<Variant> vs = {vpoly4, vpoly6, vbs4, vbs6, vbs4old};

    std::filesystem::create_directories(project_root() / "experiments" / "results");
    const auto out_dir = project_root() / "experiments" / "results";
    {
        std::ofstream csv(out_dir / "cond.csv");
        csv.precision(17);
        csv << "family,m,cond\n";
        experiment1(dat, csv);
    }
    experiment2(dat);
    const auto Y = make_Y(dat.D, 3000);
    {
        std::ofstream csv(out_dir / "fits.csv");
        csv.precision(17);
        csv << "variant,iters,rel_err,min_eig,nInc\n";
        experiment3(dat, vs, Y, csv);
    }
    {
        std::ofstream csv(out_dir / "ccp.csv");
        csv.precision(17);
        csv << "N,method,rounds,f0,f1,min_eig\n";
        experiment4(vpoly4, vbs4, Y, csv);
    }
    {
        std::ofstream csv(out_dir / "nsweep.csv");
        csv.precision(17);
        csv << "N,variant,iters,rel_err,min_eig,nInc\n";
        experiment5(dat, vs, vpoly4, vbs4, csv);
    }
    return 0;
}
