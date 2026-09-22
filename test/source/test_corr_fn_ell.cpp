// -*- coding: utf-8 -*-
#include <doctest/doctest.h>

#include <corrsolver/corr_solver.hpp>
#include <corrsolver/sites.hpp>
#include <cstddef>
#include <ellalgo/arr.hpp>

TEST_CASE("check create_2d_isotropic") {
    const auto site = create_2d_sites(5, 4);
    const auto Y = create_2d_isotropic(site, 3000);
    CHECK_EQ(site(6, 0), doctest::Approx(2.5));
    CHECK_EQ(Y.rows(), 20);
    CHECK_EQ(Y.cols(), 20);
}

TEST_CASE("lsq_corr_fn") {
    const auto site = create_2d_sites(10, 8);
    const auto Y = create_2d_isotropic(site, 3000);
    const auto fit = lsq_corr_poly2(Y, site, 4);
    REQUIRE(fit.ok);
    CHECK_GE(fit.coeffs[0], 0.0);
    CHECK_GE(fit.iters, 440);
    CHECK_LE(fit.iters, 1100);
}

TEST_CASE("mle_corr_fn") {
    const auto site = create_2d_sites(10, 8);
    const auto Y = create_2d_isotropic(site, 3000);
    const auto fit = mle_corr_poly(Y, site, 4);
    REQUIRE(fit.ok);
    CHECK_GE(fit.coeffs[0], 0.0);
    CHECK_GE(fit.iters, 50);
    CHECK_LE(fit.iters, 500);
}

TEST_CASE("cccp_corr_fn") {
    const auto site = create_2d_sites(10, 8);
    const auto Y = create_2d_isotropic(site, 3000);
    const auto fit = cccp_corr_poly(Y, site, 4);
    REQUIRE(fit.ok);
    CHECK_GE(fit.coeffs[0], 0.0);
    CHECK_GE(fit.iters, 1U);
}

TEST_CASE("cccp_matches_mle_when_constraint_inactive") {
    const auto site = create_2d_sites(10, 8);
    const auto Y = create_2d_isotropic(site, 3000);
    const auto res_mle = mle_corr_poly(Y, site, 4);
    const auto res_cccp = cccp_corr_poly(Y, site, 4);
    REQUIRE(res_mle.coeffs.size() == res_cccp.coeffs.size());
    for (size_t i = 0; i < res_mle.coeffs.size(); ++i)
        CHECK(res_cccp.coeffs(i) == doctest::Approx(res_mle.coeffs(i)).epsilon(1e-2));
}
