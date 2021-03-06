// -*- coding: utf-8 -*-
#include <algorithm>                    // for copy
#include <cmath>                        // for sqrt, exp
#include <corrsolver/Qmi_oracle.hpp>    // for Qmi_oracle
#include <cstddef>                      // for size_t
#include <ellalgo/cut_config.hpp>       // for CInfo
#include <ellalgo/cutting_plane.hpp>    // for cutting_plane_dc, bsearch
#include <ellalgo/ell.hpp>              // for ell
#include <gsl/span>                     // for span
#include <lmisolver/ldlt_ext.hpp>       // for ldlt_ext
#include <lmisolver/lmi0_oracle.hpp>    // for lmi0_oracle
#include <lmisolver/lmi_oracle.hpp>     // for lmi_oracle, lmi_oracle::Arr
#include <optional>                     // for optional
#include <tuple>                        // for tuple_element<>::type
#include <tuple>                        // for tuple, make_tuple
#include <type_traits>                  // for move, add_const<>::type
#include <utility>                      // for make_pair, pair
#include <vector>                       // for vector, __vector_base<>::v...
#include <xtensor-blas/xlinalg.hpp>     // for dot, trace, cholesky, inv
#include <xtensor/xaccessible.hpp>      // for xconst_accessible
#include <xtensor/xarray.hpp>           // for xarray_container
#include <xtensor/xbroadcast.hpp>       // for xbroadcast
#include <xtensor/xbuilder.hpp>         // for linspace, ones, diagonal
#include <xtensor/xcontainer.hpp>       // for xcontainer, xcontainer<>::...
#include <xtensor/xfunction.hpp>        // for xfunction
#include <xtensor/xgenerator.hpp>       // for xgenerator
#include <xtensor/xiterator.hpp>        // for operator==, linear_begin
#include <xtensor/xlayout.hpp>          // for layout_type, layout_type::...
#include <xtensor/xmanipulation.hpp>    // for transpose, flatten
#include <xtensor/xmath.hpp>            // for log, sum, log_fun
#include <xtensor/xoperation.hpp>       // for xfunction_type_t, operator*
#include <xtensor/xrandom.hpp>          // for default_engine_type, randn
#include <xtensor/xreducer.hpp>         // for xreducer
#include <xtensor/xsemantic.hpp>        // for xsemantic_base
#include <xtensor/xslice.hpp>           // for all, range
#include <xtensor/xstrided_view.hpp>    // for xstrided_view
#include <xtensor/xtensor.hpp>          // for xtensor_container
#include <xtensor/xtensor_forward.hpp>  // for xarray
#include <xtensor/xview.hpp>            // for xview, view
#include <xtl/xiterator_base.hpp>       // for operator!=

using Arr = xt::xarray<double, xt::layout_type::row_major>;

/*!
 * @brief Create a 2d sites object
 *
 * @param[in] nx
 * @param[in] ny
 * @return Arr location of sites
 */
Arr create_2d_sites(size_t nx = 10U, size_t ny = 8U) {
    // const auto n = nx * ny;
    const auto s_end = Arr{10.0, 8.0};
    const auto sx = xt::linspace<double>(0.0, s_end[0], nx);
    const auto sy = xt::linspace<double>(0.0, s_end[1], ny);
    const auto [xx, yy] = xt::meshgrid(sx, sy);
    const auto st = Arr{xt::stack(xt::xtuple(xt::flatten(xx), xt::flatten(yy)), 0)};
    return xt::transpose(st);
}

/*!
 * @brief Create a 2d isotropic object
 *
 * @param[in] s location of sites
 * @param[in] N
 * @return Arr Biased covariance matrix
 */
Arr create_2d_isotropic(const Arr& s, size_t N = 3000U) {
    using xt::linalg::dot;

    const auto n = s.shape()[0];
    const auto sdkern = 0.3;   // width of kernel
    const auto var = 2.;       // standard derivation
    const auto tau = 0.00001;  // standard derivation of white noise
    xt::random::seed(5);

    Arr Sig = xt::zeros<double>({n, n});
    for (auto i = 0U; i != n; ++i) {
        for (auto j = i; j != n; ++j) {
            auto d = xt::view(s, j, xt::all()) - xt::view(s, i, xt::all());
            auto g = -sdkern * std::sqrt(dot(d, d)());
            Sig(i, j) = std::exp(g);
            Sig(j, i) = Sig(i, j);
        }
    }

    auto A = xt::linalg::cholesky(Sig);
    Arr Y = xt::zeros<double>({n, n});
    for (auto k = 0U; k != N; ++k) {
        auto x = var * xt::random::randn<double>({n});
        auto y = dot(A, x) + tau * xt::random::randn<double>({n});
        // Arr y = dot(A, x);
        Y += xt::linalg::outer(y, y);
    }
    Y /= N;

    return Y;
}

/*!
 * @brief Construct a distance matrix object
 *
 * @param[in] s location of sites
 * @return std::vector<Arr>
 */
Arr construct_distance_matrix(const Arr& s) {
    auto n = s.shape()[0];
    Arr D1 = xt::zeros<double>({n, n});
    for (auto i = 0U; i != n; ++i) {
        for (auto j = i + 1; j != n; ++j) {
            auto h = xt::view(s, j, xt::all()) - xt::view(s, i, xt::all());
            auto d = std::sqrt(xt::linalg::dot(h, h)());
            D1(i, j) = d;
            D1(j, i) = d;
        }
    }
    return D1;
}

/*!
 * @brief Construct distance matrix for polynomial
 *
 * @param[in] s location of sites
 * @param[in] m degree of polynomial
 * @return std::vector<Arr>
 */
std::vector<Arr> construct_poly_matrix(const Arr& s, size_t m) {
    auto n = s.shape()[0];
    auto D1 = construct_distance_matrix(s);
    auto D = Arr{xt::ones<double>({n, n})};
    auto Sig = std::vector<Arr>{D};
    Sig.reserve(m);

    for (auto i = 0U; i != m - 1; ++i) {
        D *= D1;
        Sig.push_back(D);
    }
    return Sig;
}

/*!
 * @brief
 *
 *    min   || \Sigma(p) - Y ||
 *    s.t.  \Sigma(p) >= 0
 *
 *    where
 *
 *        \rho(h) = p1 \phi1(h) + ... + pn \phin(h)
 *
 *        {Fk}i,j = \phik( ||sj - si||^2 )
 */
class lsq_oracle {
    using Arr = xt::xarray<double, xt::layout_type::row_major>;
    using shape_type = Arr::shape_type;
    using Cut = std::tuple<Arr, double>;

  private:
    Qmi_oracle<Arr> _qmi;
    lmi0_oracle<Arr> _lmi0;

  public:
    /*!
     * @brief Construct a new lsq oracle object
     *
     * @param[in] F
     * @param[in] F0
     */
    lsq_oracle(const std::vector<Arr>& F, const Arr& F0) : _qmi(F, F0), _lmi0(F) {}

    /*!
     * @brief
     *
     * @param[in] x
     * @param[in] t the best-so-far optimal value
     * @return auto
     */
    std::tuple<Cut, bool> operator()(const Arr& x, double& t) {
        const auto n = x.size();
        Arr g = xt::zeros<double>({n});
        if (const auto cut0 = this->_lmi0(xt::view(x, xt::range(0, n - 1)))) {
            const auto& [g0, f0] = *cut0;
            xt::view(g, xt::range(0, n - 1)) = g0;
            g(n - 1) = 0.;
            return {{std::move(g), f0}, false};
        }
        this->_qmi.update(x(n - 1));

        if (const auto cut1 = this->_qmi(xt::view(x, xt::range(0, n - 1)))) {
            const auto& [g1, f1] = *cut1;
            const auto& Q = this->_qmi._Q;
            const auto& [start, stop] = Q.p;
            const auto v = xt::view(Q.witness_vec, xt::range(start, stop));
            xt::view(g, xt::range(0, n - 1)) = g1;
            g(n - 1) = -xt::linalg::dot(v, v)();
            return {{std::move(g), f1}, false};
        }
        g(n - 1) = 1.;

        const auto f0 = x(n - 1) - t;
        if (f0 > 0) {
            return {{std::move(g), f0}, false};
        }

        t = x(n - 1);
        return {{std::move(g), 0.0}, true};
    }
};

/*!
 * @brief
 *
 * @param[in] Y
 * @param[in] m
 * @param[in] P
 * @return auto
 */
auto lsq_corr_core2(const Arr& Y, size_t m, lsq_oracle& P) {
    auto normY = 100.0 * xt::linalg::norm(Y);
    auto normY2 = 32.0 * normY * normY;
    auto val = Arr{256.0 * xt::ones<double>({m + 1})};

    val(m) = normY2 * normY2;
    Arr x = xt::zeros<double>({m + 1});
    x(0) = 4;
    x(m) = normY2 / 2.;
    auto E = ell(val, x);
    auto t = 1e100;  // std::numeric_limits<double>::max()
    const auto [x_best, ell_info] = cutting_plane_dc(P, E, t);
    Arr a = xt::view(x_best, xt::range(0, m));
    return std::make_tuple(std::move(a), ell_info.num_iters, ell_info.feasible);
}

/*!
 * @brief
 *
 * @param[in] Y
 * @param[in] s
 * @param[in] m
 * @return std::tuple<size_t, bool>
 */
std::tuple<Arr, size_t, bool> lsq_corr_poly2(const Arr& Y, const Arr& s, size_t m) {
    auto Sig = construct_poly_matrix(s, m);
    auto P = lsq_oracle(Sig, Y);
    return lsq_corr_core2(Y, m, P);
}

/*!
 * @brief
 *
 */
class mle_oracle {
    using Arr = xt::xarray<double, xt::layout_type::row_major>;
    using shape_type = Arr::shape_type;
    using Cut = std::tuple<Arr, double>;

  private:
    const Arr& _Y;
    const std::vector<Arr>& _Sig;
    lmi0_oracle<Arr> _lmi0;
    lmi_oracle<Arr> _lmi;

  public:
    /*!
     * @brief Construct a new mle oracle object
     *
     * @param[in] Sig
     * @param[in] Y
     */
    mle_oracle(const std::vector<Arr>& Sig, const Arr& Y)
        : _Y{Y}, _Sig{Sig}, _lmi0(Sig), _lmi(Sig, 2.0 * Y) {}

    /*!
     * @brief
     *
     * @param[in] x
     * @param[in] t the best-so-far optimal value
     * @return auto
     */
    std::tuple<Cut, bool> operator()(const Arr& x, double& t) {
        using xt::linalg::dot;

        const auto cut1 = this->_lmi(x);
        if (cut1) {
            return {*cut1, false};
        }

        const auto cut0 = this->_lmi0(x);
        if (cut0) {
            return {*cut0, false};
        }

        auto n = x.shape()[0];
        auto m = this->_Y.shape()[0];

        const auto& R = this->_lmi0._Q.sqrt();
        auto invR = Arr{xt::linalg::inv(R)};
        auto S = Arr{dot(invR, xt::transpose(invR))};
        auto SY = Arr{dot(S, this->_Y)};

        auto diag = xt::diagonal(R);
        auto f1 = double{2.0 * xt::sum(xt::log(diag))() + xt::linalg::trace(SY)()};
        // auto f1 = 0.;

        auto f = f1 - t;
        auto shrunk = false;

        if (f < 0) {
            t = f1;
            f = 0.;
            shrunk = true;
        }

        Arr g = xt::zeros<double>({n});

        for (auto i = 0U; i != n; ++i) {
            auto SFsi = dot(S, this->_Sig[i]);
            g(i) = xt::linalg::trace(SFsi)();
            for (auto k = 0U; k != m; ++k) {
                g(i) -= dot(xt::view(SFsi, k, xt::all()), xt::view(SY, xt::all(), k))();
            }
        }
        return {{std::move(g), f}, shrunk};
    }
};

/*!
 * @brief
 *
 * @param[in] Y
 * @param[in] m
 * @param[in] P
 * @return auto
 */
auto mle_corr_core(const Arr& /* Y */, size_t m, mle_oracle& P) {
    Arr x = xt::zeros<double>({m});
    x(0) = 4.;
    auto E = ell(500.0, x);
    auto t = 1e100;  // std::numeric_limits<double>::max()
    auto [x_best, ell_info] = cutting_plane_dc(P, E, t);
    return std::make_tuple(std::move(x_best), ell_info.num_iters, ell_info.feasible);
}

/*!
 * @brief
 *
 * @param[in] Y
 * @param[in] s
 * @param[in] m
 * @return std::tuple<size_t, bool>
 */
std::tuple<Arr, size_t, bool> mle_corr_poly(const Arr& Y, const Arr& s, size_t m) {
    const auto Sig = construct_poly_matrix(s, m);
    auto P = mle_oracle(Sig, Y);
    return mle_corr_core(Y, m, P);
}

/*!
 * @brief
 *
 * @param[in] Y
 * @param[in] s
 * @param[in] m
 * @return std::tuple<size_t, bool>
 */
std::tuple<Arr, size_t, bool> lsq_corr_poly(const Arr& Y, const Arr& s, size_t m) {
    auto Sig = construct_poly_matrix(s, m);
    // P = mtx_norm_oracle(Sig, Y, a)
    auto a = xt::zeros<double>({m});
    auto Q = Qmi_oracle<Arr>(Sig, Y);
    auto E = ell(10.0, a);
    auto P = bsearch_adaptor<decltype(Q), decltype(E)>(Q, E);
    // double normY = xt::norm_l2(Y);
    auto bs_info = bsearch(P, std::make_pair(0.0, 100.0 * 100.0));

    // std::cout << niter << ", " << feasible << '\n';
    return {P.x_best(), bs_info.num_iters, bs_info.feasible};
    //  return prob.is_dcp()
}
