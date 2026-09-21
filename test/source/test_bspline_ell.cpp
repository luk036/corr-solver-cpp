// -*- coding: utf-8 -*-
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <corrsolver/bspline.hpp>
#include <corrsolver/eigen.hpp>
#include <corrsolver/halton.hpp>
#include <cstddef>
#include <ellalgo/arr.hpp>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

extern std::vector<Arr> construct_poly_matrix(const Arr&, size_t);

namespace {

double dmax_of(const Arr& D) {
    double dmax = 0.0;
    for (size_t i = 0; i < D.size(); ++i) dmax = std::max(dmax, D(i));
    return dmax;
}

double d_span_of(const Arr& site) {
    double s = 0.0;
    for (size_t j = 0; j < site.cols(); ++j) {
        double diff = site(site.rows() - 1, j) - site(0, j);
        s += diff * diff;
    }
    return std::sqrt(s);
}

Arr legacy_knots(double d_span, size_t m, size_t k) {
    return linspace(0.0, 1.2 * d_span, m + k + 1);
}

struct MockBasis {
    bool called = false;
    std::tuple<Cut, bool> assess_optim(const Arr& x, double& t) {
        called = true;
        (void)t;
        return {{zeros(x.size()), 0.0}, true};
    }
};

}

TEST_CASE("halton sites match reference rows") {
    const auto site = create_2d_sites_halton(5, 4);
    REQUIRE(site.rows() == 20);
    REQUIRE(site.cols() == 2);
    CHECK(site(0, 0) == doctest::Approx(5.0).epsilon(1e-12));
    CHECK(site(0, 1) == doctest::Approx(2.6666666666666665).epsilon(1e-12));
    CHECK(site(6, 0) == doctest::Approx(8.75).epsilon(1e-12));
    CHECK(site(6, 1) == doctest::Approx(4.444444444444445).epsilon(1e-12));
    CHECK(site(19, 0) == doctest::Approx(1.5625).epsilon(1e-12));
    CHECK(site(19, 1) == doctest::Approx(5.925925925925926).epsilon(1e-12));
}

TEST_CASE("distance matrix dmax and d_span") {
    const auto site = create_2d_sites_halton(5, 4);
    const auto D = construct_distance_matrix(site);
    CHECK(dmax_of(D) == doctest::Approx(10.096248912961826).epsilon(1e-12));
    CHECK(d_span_of(site) == doctest::Approx(4.737000862261608).epsilon(1e-12));
}

TEST_CASE("clamped knots m=4") {
    const auto site = create_2d_sites_halton(5, 4);
    const auto t = clamped_knots(dmax_of(construct_distance_matrix(site)), 4, 2);
    REQUIRE(t.size() == 7);
    const double expected[7] = {0.0,
                                0.0,
                                0.0,
                                5.048124456480913,
                                10.096248912961826,
                                10.096248912961826,
                                10.096248912961826};
    for (size_t i = 0; i < 7; ++i) CHECK(t(i) == doctest::Approx(expected[i]).epsilon(1e-12));
}

TEST_CASE("clamped basis m=4 spot values and partition of unity") {
    const auto site = create_2d_sites_halton(5, 4);
    const auto info = generate_bspline_info(site, 4);
    REQUIRE(info.Sigma.size() == 4);
    CHECK(info.k == 2);
    const auto& B0 = info.Sigma[0];
    const auto& B3 = info.Sigma[3];
    CHECK(B0(0, 1) == doctest::Approx(0.07612753830941012).epsilon(1e-12));
    CHECK(B0(0, 19) == doctest::Approx(0.0037984445216406814).epsilon(1e-12));
    CHECK(B3(0, 1) == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(B3(0, 19) == doctest::Approx(0.0).epsilon(1e-12));
    double s = 0.0;
    for (size_t k = 0; k < 4; ++k) s += info.Sigma[k](0, 19);
    CHECK(s == doctest::Approx(1.0).epsilon(1e-12));
}

TEST_CASE("legacy uniform knots reproduce scipy extrapolation") {
    const auto site = create_2d_sites_halton(5, 4);
    const auto D = construct_distance_matrix(site);
    const auto t = legacy_knots(d_span_of(site), 4, 2);
    REQUIRE(t.size() == 7);
    const double expected[7] = {0.0,
                                0.9474001724523217,
                                1.8948003449046433,
                                2.842200517356965,
                                3.7896006898092867,
                                4.737000862261608,
                                5.68440103471393};
    for (size_t i = 0; i < 7; ++i) CHECK(t(i) == doctest::Approx(expected[i]).epsilon(1e-12));
    const auto B = eval_bspline_basis(t, 2, D);
    REQUIRE(B.size() == 4);
    CHECK(B[0](0, 19) == doctest::Approx(0.0).epsilon(1e-12));
    CHECK(B[3](0, 19) == doctest::Approx(2.0).epsilon(1e-12));
}

TEST_CASE("generate_bspline_info rejects too few control points") {
    const auto site = create_2d_sites_halton(5, 4);
    CHECK_THROWS_AS(generate_bspline_info(site, 2), std::invalid_argument);
}

TEST_CASE("design cond poly") {
    const auto site = create_2d_sites_halton(5, 4);
    struct Case {
        size_t m;
        double ref;
        double eps;
    };
    const Case cases[] = {{2, 11.59155345705951, 1e-9},
                          {4, 1210.3830070574695, 1e-9},
                          {6, 187488.94022165914, 1e-9},
                          {8, 67217446.67806517, 1e-9},
                          // m=10: np.linalg.cond's smallest singular value carries
                          // ~eps*cond error, placing the reference ~3e-9 below the
                          // double-double result.
                          {10, 34546585337.19029, 1e-8}};
    for (const auto& c : cases)
        CHECK(design_cond(construct_poly_matrix(site, c.m)) ==
              doctest::Approx(c.ref).epsilon(c.eps));
}

TEST_CASE("design cond clamped bspline") {
    const auto site = create_2d_sites_halton(5, 4);
    struct Case {
        size_t m;
        double ref;
    };
    const Case cases[] = {{4, 5.052018587714944},
                          {6, 6.478335493279365},
                          {8, 6.507672636371483},
                          {10, 6.751181601819613}};
    for (const auto& c : cases) {
        const auto info = generate_bspline_info(site, c.m);
        CHECK(design_cond(info.Sigma) == doctest::Approx(c.ref).epsilon(1e-9));
    }
}

TEST_CASE("design cond legacy uniform bspline") {
    const auto site = create_2d_sites_halton(5, 4);
    const auto D = construct_distance_matrix(site);
    const auto ds = d_span_of(site);
    struct Case {
        size_t m;
        double ref;
    };
    const Case cases[] = {{4, 40.26126074981864},
                          {6, 99.40044633714751},
                          {8, 203.56551712290315},
                          {10, 409.4721392889807}};
    for (const auto& c : cases) {
        const auto t = legacy_knots(ds, c.m, 2);
        const auto Sig = eval_bspline_basis(t, 2, D);
        CHECK(design_cond(Sig) == doctest::Approx(c.ref).epsilon(1e-9));
    }
}

TEST_CASE("design cond on a controlled ill-conditioned matrix") {
    const double e = 1e-10;
    const double th = 0.7;
    const double c = std::cos(th);
    const double s = std::sin(th);
    std::vector<Arr> Sig(2);
    Sig[0] = Arr(3);
    Sig[0](0) = c;
    Sig[0](1) = -e * s;
    Sig[0](2) = 0.0;
    Sig[1] = Arr(3);
    Sig[1](0) = s;
    Sig[1](1) = e * c;
    Sig[1](2) = 0.0;
    CHECK(design_cond(Sig) == doctest::Approx(1.0 / e).epsilon(1e-9));
}

TEST_CASE("design cond on a controlled m=10 matrix") {
    const double pi = 3.14159265358979323846;
    const size_t N = 10;
    std::vector<double> d(N);
    d[0] = 1e-20;
    for (size_t i = 1; i < N; ++i) d[i] = static_cast<double>(i);
    std::vector<Arr> Sig(N);
    for (size_t k = 0; k < N; ++k) {
        Sig[k] = Arr(N);
        for (size_t i = 0; i < N; ++i) {
            double q = (k == 0) ? std::sqrt(1.0 / N)
                                : std::sqrt(2.0 / N) * std::cos(pi * (2.0 * i + 1.0) * k / (2.0 * N));
            Sig[k](i) = std::sqrt(d[i]) * q;
        }
    }
    CHECK(design_cond(Sig) == doctest::Approx(3e10).epsilon(1e-9));
}

TEST_CASE("jacobi eigvals and min_eig") {
    Arr M(2, 2);
    M(0, 0) = 2.0;
    M(0, 1) = 1.0;
    M(1, 0) = 1.0;
    M(1, 1) = 2.0;
    const auto ev = jacobi_eigvals(M);
    REQUIRE(ev.size() == 2);
    CHECK(ev[0] == doctest::Approx(1.0).epsilon(1e-12));
    CHECK(ev[1] == doctest::Approx(3.0).epsilon(1e-12));

    Arr D(3, 3);
    D(0, 0) = 3.0;
    D(1, 1) = 1.0;
    D(2, 2) = 2.0;
    CHECK(min_eig(D) == doctest::Approx(1.0).epsilon(1e-12));
}

TEST_CASE("mono oracle returns first violation") {
    const Arr dec = {3.0, 2.0, 1.0};
    CHECK_FALSE(mono_oracle(dec).has_value());
    const Arr inc = {1.0, 2.0, 0.5};
    const auto cut = mono_oracle(inc);
    REQUIRE(cut.has_value());
    CHECK(cut->first(0) == doctest::Approx(-1.0));
    CHECK(cut->first(1) == doctest::Approx(1.0));
    CHECK(cut->first(2) == doctest::Approx(0.0));
    CHECK(cut->second == doctest::Approx(1.0));
}

TEST_CASE("mono decreasing oracle2 delegates and constrains leading coeffs") {
    MockBasis basis;
    MonoDecreasingOracle2<MockBasis> oracle(basis);
    const Arr x = {3.0, 2.0, 1.0, 0.5};
    double t = 1.0;
    const auto res = oracle.assess_optim(x, t);
    CHECK(std::get<1>(res));
    CHECK(basis.called);

    MockBasis basis2;
    MonoDecreasingOracle2<MockBasis> oracle2(basis2, std::optional<size_t>(2));
    const Arr y = {1.0, 2.0, 5.0, 6.0};
    double t2 = 1.0;
    const auto res2 = oracle2.assess_optim(y, t2);
    CHECK_FALSE(std::get<1>(res2));
    CHECK_FALSE(basis2.called);
    const auto& cut2 = std::get<0>(res2);
    CHECK(cut2.first.size() == 4);
    CHECK(cut2.first(0) == doctest::Approx(-1.0));
    CHECK(cut2.first(1) == doctest::Approx(1.0));
    CHECK(cut2.first(2) == doctest::Approx(0.0));
    CHECK(cut2.second == doctest::Approx(1.0));

    MockBasis basis3;
    MonoDecreasingOracle2<MockBasis> oracle3(basis3, std::optional<size_t>(2));
    const Arr z = {2.0, 1.0, 5.0, 6.0};
    double t3 = 1.0;
    const auto res3 = oracle3.assess_optim(z, t3);
    CHECK(std::get<1>(res3));
    CHECK(basis3.called);
}
