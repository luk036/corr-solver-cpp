// -*- coding: utf-8 -*-
#include <doctest/doctest.h>

#include <cstddef>
#include <ellalgo/arr.hpp>
#include <tuple>

extern Arr create_2d_sites(size_t, size_t);
extern Arr create_2d_isotropic(const Arr&, size_t);
extern std::tuple<Arr, size_t> lsq_corr_poly2(const Arr&, const Arr&, size_t);
extern std::tuple<Arr, size_t> mle_corr_poly(const Arr&, const Arr&, size_t);
extern std::tuple<Arr, size_t> cccp_corr_poly(const Arr&, const Arr&, size_t);

TEST_CASE("check create_2d_isotropic") {
    const auto site = create_2d_sites(5, 4);
    const auto Y = create_2d_isotropic(site, 3000);
    CHECK_EQ(site(6, 0), doctest::Approx(2.5));
}

TEST_CASE("lsq_corr_fn") {
    const auto site = create_2d_sites(10, 8);
    const auto Y = create_2d_isotropic(site, 3000);
    const auto [coeffs, num_iters] = lsq_corr_poly2(Y, site, 4);
    REQUIRE(coeffs.size() > 0);
    CHECK_GE(coeffs[0], 0.0);
    CHECK_GE(num_iters, 440);
    CHECK_LE(num_iters, 1100);
}

TEST_CASE("mle_corr_fn") {
    const auto site = create_2d_sites(10, 8);
    const auto Y = create_2d_isotropic(site, 3000);
    const auto [coeffs, num_iters] = mle_corr_poly(Y, site, 4);
    REQUIRE(coeffs.size() > 0);
    CHECK_GE(coeffs[0], 0.0);
    CHECK_GE(num_iters, 50);
    CHECK_LE(num_iters, 500);
}

TEST_CASE("cccp_corr_fn") {
    const auto site = create_2d_sites(10, 8);
    const auto Y = create_2d_isotropic(site, 3000);
    const auto result = cccp_corr_poly(Y, site, 4);
    const auto& coeffs = std::get<0>(result);
    const auto num_iters = std::get<1>(result);
    REQUIRE(coeffs.size() > 0);
    CHECK_GE(coeffs[0], 0.0);
    CHECK_GE(num_iters, 1U);
}

TEST_CASE("cccp_matches_mle_when_constraint_inactive") {
    const auto site = create_2d_sites(10, 8);
    const auto Y = create_2d_isotropic(site, 3000);
    const auto res_mle = mle_corr_poly(Y, site, 4);
    const auto res_cccp = cccp_corr_poly(Y, site, 4);
    const auto& c_mle = std::get<0>(res_mle);
    const auto& c_cccp = std::get<0>(res_cccp);
    REQUIRE(c_mle.size() == c_cccp.size());
    for (size_t i = 0; i < c_mle.size(); ++i)
        CHECK(c_cccp(i) == doctest::Approx(c_mle(i)).epsilon(1e-2));
}
