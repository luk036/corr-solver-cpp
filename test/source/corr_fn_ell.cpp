// -*- coding: utf-8 -*-
#include <doctest/doctest.h>  // for ResultBuilder, CHECK, TestCase
#include <stddef.h>           // for size_t

#include <tuple>                        // for tuple
#include <xtensor/xaccessible.hpp>      // for xconst_accessible
#include <xtensor/xarray.hpp>           // for xarray_container
#include <xtensor/xcontainer.hpp>       // for xcontainer
#include <xtensor/xlayout.hpp>          // for layout_type, layout_type::row...
#include <xtensor/xtensor_forward.hpp>  // for xarray

using Arr = xt::xarray<double, xt::layout_type::row_major>;

extern Arr create_2d_sites(size_t, size_t);
extern Arr create_2d_isotropic(const Arr &, size_t);
extern std::tuple<Arr, size_t> lsq_corr_poly2(const Arr &, const Arr &, size_t);
extern std::tuple<Arr, size_t> mle_corr_poly(const Arr &, const Arr &, size_t);

TEST_CASE("check create_2d_isotropic") {
    const auto site = create_2d_sites(5, 4);
    const auto Y = create_2d_isotropic(site, 3000);
    CHECK(site(6, 0) == doctest::Approx(2.5));
}

TEST_CASE("lsq_corr_fn") {
    const auto site = create_2d_sites(10, 8);
    const auto Y = create_2d_isotropic(site, 3000);
    const auto [coeffs, num_iters] = lsq_corr_poly2(Y, site, 4);
    REQUIRE(coeffs.size() > 0);
    CHECK(coeffs[0] >= 0.0);
    CHECK(num_iters >= 440);
    CHECK(num_iters <= 723);
}

TEST_CASE("mle_corr_fn") {
    const auto site = create_2d_sites(10, 8);
    const auto Y = create_2d_isotropic(site, 3000);
    const auto [coeffs, num_iters] = mle_corr_poly(Y, site, 4);
    REQUIRE(coeffs.size() > 0);
    CHECK(coeffs[0] >= 0.0);
    CHECK(num_iters >= 227);
    CHECK(num_iters <= 247);
}
