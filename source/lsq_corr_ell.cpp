// -*- coding: utf-8 -*-
#include <algorithm>                        // for copy
#include <cmath>                            // for sqrt, exp
#include <corrsolver/qmi_oracle.hpp>        // for QmiOracle
#include <cstddef>                          // for size_t
#include <ellalgo/cutting_plane.hpp>        // for cutting_plane_optim, bsearch
#include <ellalgo/ell.hpp>                  // for Ell
#include <ellalgo/oracles/ldlt_mgr.hpp>     // for LDLTMgr
#include <ellalgo/oracles/lmi0_oracle.hpp>  // for Lmi0Oracle
#include <ellalgo/oracles/lmi_oracle.hpp>   // for LmiOracle, LmiOracle::Arr
#include <optional>                         // for optional
#include <tuple>                            // for tuple_element<>::type
#include <tuple>                            // for tuple, make_tuple
#include <type_traits>                      // for move, add_const<>::type
#include <utility>                          // for make_pair, pair
#include <vector>                           // for vector, __vector_base<>::v...
#include <xtensor-blas/xlinalg.hpp>         // for dot, trace, cholesky, inv
#include <xtensor/xaccessible.hpp>          // for xconst_accessible
#include <xtensor/xarray.hpp>               // for xarray_container
#include <xtensor/xbroadcast.hpp>           // for xbroadcast
#include <xtensor/xbuilder.hpp>             // for linspace, ones, diagonal
#include <xtensor/xcontainer.hpp>           // for xcontainer, xcontainer<>::...
#include <xtensor/xfunction.hpp>            // for xfunction
#include <xtensor/xgenerator.hpp>           // for xgenerator
#include <xtensor/xiterator.hpp>            // for operator==, linear_begin
#include <xtensor/xlayout.hpp>              // for layout_type, layout_type::...
#include <xtensor/xmanipulation.hpp>        // for transpose, flatten
#include <xtensor/xmath.hpp>                // for log, sum, log_fun
#include <xtensor/xoperation.hpp>           // for xfunction_type_t, operator*
#include <xtensor/xrandom.hpp>              // for default_engine_type, randn
#include <xtensor/xreducer.hpp>             // for xreducer
#include <xtensor/xsemantic.hpp>            // for xsemantic_base
#include <xtensor/xslice.hpp>               // for all, range
#include <xtensor/xstrided_view.hpp>        // for xstrided_view
#include <xtensor/xtensor.hpp>              // for xtensor_container
#include <xtensor/xtensor_forward.hpp>      // for xarray
#include <xtensor/xview.hpp>                // for xview, view
#include <xtl/xiterator_base.hpp>           // for operator!=

using Arr = xt::xarray<double, xt::layout_type::row_major>;

/**
 * The function creates a 2D sites object with specified dimensions.
 *
 * @param[in] nx The parameter `nx` represents the number of points in the x-direction, while `ny`
 * represents the number of points in the y-direction.
 * @param[in] ny The parameter `ny` represents the number of rows in the 2D sites object.
 *
 * @return The function `create_2d_sites` returns a 2D array (Arr) containing the locations of
 * sites.
 */
Arr create_2d_sites(size_t nx = 10U, size_t ny = 8U) {
    const auto s_end = Arr{10.0, 8.0};
    const auto sx = xt::linspace<double>(0.0, s_end[0], nx);
    const auto sy = xt::linspace<double>(0.0, s_end[1], ny);
    const auto [xx, yy] = xt::meshgrid(sx, sy);
    const auto st = Arr{xt::stack(xt::xtuple(xt::flatten(xx), xt::flatten(yy)), 0)};
    return xt::transpose(st);
}

/**
 * The function creates a 2D isotropic object by generating a biased covariance matrix based on the
 * given sites and parameters.
 *
 * @param[in] site The parameter `site` represents the location of sites. It is an input array that
 * contains the coordinates of the sites in a 2D space. Each row of the array represents the
 * coordinates of a single site.
 * @param[in] N The parameter `N` represents the number of iterations in the loop that generates the
 * object. It determines the number of samples used to compute the biased covariance matrix.
 *
 * @return The function `create_2d_isotropic` returns a 2D array `Arr` representing the biased
 * covariance matrix.
 */
Arr create_2d_isotropic(const Arr &site, size_t N = 3000U) {
    using xt::linalg::dot;

    const auto n = site.shape()[0];
    const auto sdkern = 0.3;   // width of kernel
    const auto var = 2.;       // standard derivation
    const auto tau = 0.00001;  // standard derivation of white noise
    xt::random::seed(5);

    Arr Sig = xt::zeros<double>({n, n});
    for (auto i = 0U; i != n; ++i) {
        for (auto j = i; j != n; ++j) {
            auto d = xt::view(site, j, xt::all()) - xt::view(site, i, xt::all());
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
        Y += xt::linalg::outer(y, y);
    }
    Y /= N;

    return Y;
}

/**
 * The function constructs a distance matrix based on the locations of sites.
 *
 * @param[in] site The parameter `site` is a matrix representing the locations of sites. It is of
 * type `Arr`, which is likely a typedef for a multidimensional array or matrix. The shape of `site`
 * is assumed to be
 * `(n, m)`, where `n` is the number of sites and
 *
 * @return The function `construct_distance_matrix` returns a `std::vector<Arr>`, which is a vector
 * of `Arr` objects.
 */
Arr construct_distance_matrix(const Arr &site) {
    auto n = site.shape()[0];
    Arr D1 = xt::zeros<double>({n, n});
    for (auto i = 0U; i != n; ++i) {
        for (auto j = i + 1; j != n; ++j) {
            auto h = xt::view(site, j, xt::all()) - xt::view(site, i, xt::all());
            auto d = std::sqrt(xt::linalg::dot(h, h)());
            D1(i, j) = d;
            D1(j, i) = d;
        }
    }
    return D1;
}

/**
 * The function constructs a distance matrix for a polynomial given a set of locations and the
 * degree of the polynomial.
 *
 * @param[in] site The parameter `site` represents the location of sites. It is of type `Arr`, which
 * is likely a multidimensional array or matrix. The shape of `site` is expected to be `[n, d]`,
 * where `n` is the number of sites and `d` is the dimension
 * @param[in] m The parameter `m` represents the degree of the polynomial. It determines the number
 * of matrices that will be constructed in the distance matrix.
 *
 * @return The function `construct_poly_matrix` returns a `std::vector<Arr>`, which is a vector of
 * `Arr` objects.
 */
std::vector<Arr> construct_poly_matrix(const Arr &site, size_t m) {
    auto n = site.shape()[0];
    auto D1 = construct_distance_matrix(site);
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
 *    site.t.  \Sigma(p) >= 0
 *
 *    where
 *
 *        \rho(h) = p1 \phi1(h) + ... + pn \phin(h)
 *
 *        {Fk}i,j = \phik( ||sj - si||^2 )
 */
class LsqOracle {
    using Arr = xt::xarray<double, xt::layout_type::row_major>;
    using shape_type = Arr::shape_type;
    using Cut = std::pair<Arr, double>;

    QmiOracle<Arr> _qmi;
    Lmi0Oracle<Arr> _lmi0;

  public:
    /**
     * The function is a constructor for an LsqOracle object that takes in a size, a vector of Arr
     * objects, and an Arr object as parameters.
     *
     * @param[in] m The parameter `m` represents the number of linear matrix inequalities (LMIs) in
     * the problem.
     * @param[in] F A vector of Arr objects. It is a parameter used to construct the LsqOracle
     * object.
     * @param[in] F0 F0 is a constant vector of type Arr.
     */
    LsqOracle(size_t m, const std::vector<Arr> &F, const Arr &F0) : _qmi(F, F0), _lmi0(m, F) {}

    /**
     * The function assess_optim assesses the optimality of a given input and returns a tuple
     * containing a cut and a boolean value.
     *
     * @param x An array of values of type `Arr`.
     * @param t The parameter `t` represents the best-so-far optimal value. It is passed by
     * reference and can be modified within the function.
     *
     * @return The function `assess_optim` returns a `std::tuple` containing a `Cut` object and a
     * boolean value.
     */
    std::tuple<Cut, bool> assess_optim(const Arr &x, double &t) {
        const auto n = x.size();
        Arr g = xt::zeros<double>({n});
        auto v = xt::view(x, xt::range(0, n - 1));
        if (const auto cut0 = this->_lmi0.assess_feas(v)) {
            const auto &[g0, f0] = *cut0;
            xt::view(g, xt::range(0, n - 1)) = g0;
            g[n - 1] = 0.0;
            return {{std::move(g), f0}, false};
        }
        this->_qmi.update(x[n - 1]);

        if (const auto cut1 = this->_qmi.assess_feas(v)) {
            const auto &[g1, f1] = *cut1;
            const auto &Q = this->_qmi._mq;
            const auto &[start, stop] = Q.p;
            Arr wit_vec = xt::zeros<double>({this->_qmi._m});  // need better sol'n
            Q.set_witness_vec(wit_vec);
            const auto v2 = xt::view(wit_vec, xt::range(start, stop));
            xt::view(g, xt::range(0, n - 1)) = g1;
            g[n - 1] = -xt::linalg::dot(v2, v2)();
            return {{std::move(g), f1}, false};
        }
        g[n - 1] = 1.0;

        if (const auto f0 = x[n - 1] - t > 0) {
            return {{std::move(g), f0}, false};
        }

        t = x[n - 1];
        return {{std::move(g), 0.0}, true};
    }
};

/**
 * The function `lsq_corr_core2` performs least squares correlation using a cutting plane
 * optimization algorithm.
 *
 * @param[in] Y The parameter `Y` is an input array of type `Arr` which represents the biased
 * covariance matrix.
 * @param[in] m The parameter `m` represents the number of coefficients in the linear least squares
 * problem. It determines the size of the coefficient vector `a` that will be returned by the
 * function.
 * @param[in] omega The parameter "omega" is of type "LsqOracle". It is an object that provides
 * information about the least squares problem being solved.
 *
 * @return The function `lsq_corr_core2` returns a tuple containing three elements:
 * 1. `a`: An `Arr` object, which represents an array of coefficients.
 * 2. `num_iters`: An integer, which represents the number of iterations performed during the
 * optimization process.
 * 3. A boolean value indicating whether the size of `x_best` is not equal to 0.
 */
auto lsq_corr_core2(const Arr &Y, size_t m, LsqOracle &omega) {
    auto normY = 100.0 * xt::linalg::norm(Y);
    auto normY2 = 32.0 * normY * normY;
    std::valarray<double> val(256.0, m + 1);
    val[m] = normY2 * normY2;
    Arr x = xt::zeros<double>({m + 1});
    x[0] = 4;
    x[m] = normY2 / 2.;
    auto ellip = Ell<Arr>(val, x);
    auto t = 1e100;  // std::numeric_limits<double>::max()
    const auto [x_best, num_iters] = cutting_plane_optim(omega, ellip, t);
    // TODO: check if x_best is valid
    // TODO: make C++14 compliant: remove std::optional<>
    Arr a = xt::view(x_best, xt::range(0, m));
    return std::make_tuple(std::move(a), num_iters);
}

/**
 * The function `lsq_corr_poly2` calculates the least squares correlation for a polynomial.
 *
 * @param[in] Y The parameter `Y` is an input array of type `Arr` which represents the biased
 * covariance matrix.
 * @param site The parameter `site` represents the site or location of the observed data. It is used
 * in constructing the polynomial matrix `Sig`.
 * @param[in] m The parameter `m` represents the degree of the polynomial that will be used for the
 * least squares fitting. It determines the number of coefficients in the polynomial equation.
 *
 * @return The function `lsq_corr_poly2` returns a `std::tuple` containing three elements: an `Arr`
 * object, a `size_t` value, and a `bool` value.
 */
std::tuple<Arr, size_t> lsq_corr_poly2(const Arr &Y, const Arr &site, size_t m) {
    auto Sig = construct_poly_matrix(site, m);
    auto omega = LsqOracle(Y.shape()[0], Sig, Y);
    return lsq_corr_core2(Y, m, omega);
}

/*!
 * @brief
 *
 */
class MleOracle {
    using Arr = xt::xarray<double, xt::layout_type::row_major>;
    using shape_type = Arr::shape_type;
    using Cut = std::pair<Arr, double>;

    const Arr &_Y;
    const std::vector<Arr> &_Sig;
    Lmi0Oracle<Arr> _lmi0;
    LmiOracle<Arr> _lmi;

  public:
    /**
     * The code snippet defines a constructor for a class called MleOracle that takes in parameters
     * m, Sig, and Y and initializes private member variables _Y, _Sig, _lmi0, and _lmi.
     *
     * @param[in] m The parameter `m` represents the size of the problem or the number of variables
     * in the problem.
     * @param[in] Sig A vector of Arr objects representing the input signals.
     * @param[in] Y The parameter `Y` is an input array of type `Arr` which represents the biased
     * covariance matrix.
     */
    MleOracle(size_t m, const std::vector<Arr> &Sig, const Arr &Y)
        : _Y{Y}, _Sig{Sig}, _lmi0(m, Sig), _lmi(m, Sig, 2.0 * Y) {}

    /**
     * The function assess_optim assesses the optimality of a given input and returns a tuple
     * containing a cut and a boolean value indicating whether the input has been shrunk.
     *
     * @param[in] x The parameter `x` is of type `Arr`, which is likely a multidimensional array or
     * matrix. It represents some input data.
     * @param[in] t The parameter `t` represents the best-so-far optimal value. It is passed by
     * reference, which means its value can be modified within the function and the updated value
     * will be reflected outside the function.
     *
     * @return The function `assess_optim` returns a `std::tuple` containing two elements: a `Cut`
     * object and a boolean value.
     */
    std::tuple<Cut, bool> assess_optim(const Arr &x, double &t) {
        using xt::linalg::dot;

        if (const auto cut1 = this->_lmi.assess_feas(x)) {
            return {*cut1, false};
        }
        if (const auto cut0 = this->_lmi0.assess_feas(x)) {
            return {*cut0, false};
        }

        auto n = x.shape()[0];
        auto m = this->_Y.shape()[0];

        const auto dim = this->_lmi0._mq._n;
        Arr R = xt::zeros<double>({dim, dim});
        this->_lmi0._mq.sqrt(R);
        auto invR = Arr{xt::linalg::inv(R)};
        auto S = Arr{dot(invR, xt::transpose(invR))};
        auto SY = Arr{dot(S, this->_Y)};

        auto diag = xt::diagonal(R);
        auto f1 = double{2.0 * xt::sum(xt::log(diag))() + xt::linalg::trace(SY)()};
        auto f = f1 - t;
        auto shrunk = false;

        if (f < 0.0) {
            t = f1;
            f = 0.0;
            shrunk = true;
        }

        Arr g = xt::zeros<double>({n});

        for (auto i = 0U; i != n; ++i) {
            auto SFsi = dot(S, this->_Sig[i]);
            g[i] = xt::linalg::trace(SFsi)();
            for (auto k = 0U; k != m; ++k) {
                g[i] -= dot(xt::view(SFsi, k, xt::all()), xt::view(SY, xt::all(), k))();
            }
        }
        return {{std::move(g), f}, shrunk};
    }
};

/**
 * The function `mle_corr_core` performs maximum likelihood estimation for correlation coefficients
 * using a cutting plane optimization algorithm.
 *
 * @param m The parameter `m` represents the size of the array `x`. It is used to create an array of
 * size `m` and initialize it with zeros.
 * @param omega The parameter `omega` is of type `MleOracle&`. It is a reference to an object of
 * type `MleOracle`, which is likely a class or struct that provides some functionality related to
 * maximum likelihood estimation (MLE). The `mle_corr_core` function uses this `omega`
 *
 * @return The function `mle_corr_core` returns a tuple containing three elements:
 * 1. `x_best`: The best solution found during the optimization process.
 * 2. `num_iters`: The number of iterations performed during the optimization process.
 */
auto mle_corr_core(size_t m, MleOracle &omega) {
    Arr x = xt::zeros<double>({m});
    x[0] = 4.0;
    auto ellip = Ell<Arr>(500.0, x);
    auto t = 1e100;  // std::numeric_limits<double>::max()
    return cutting_plane_optim(omega, ellip, t);
}

/**
 * The function `mle_corr_poly` calculates the maximum likelihood estimate of a polynomial
 * correlation matrix.
 *
 * @param[in] Y The parameter `Y` represents the biased covariance matrix. It is used to
 * calculate the maximum likelihood estimate (MLE) of the correlation polynomial.
 * @param site The parameter `site` represents the site or location of the observed data. It is used
 * in constructing the polynomial matrix `Sig`.
 * @param m The parameter `m` represents the degree of the polynomial used in constructing the
 * polynomial matrix `Sig`. It determines the number of columns in the matrix.
 *
 * @return The function `mle_corr_poly` returns a `std::tuple` containing three elements: an `Arr`
 * object, a `size_t` value, and a `bool` value.
 */
std::tuple<Arr, size_t> mle_corr_poly(const Arr &Y, const Arr &site, size_t m) {
    const auto Sig = construct_poly_matrix(site, m);
    auto omega = MleOracle(Y.shape()[0], Sig, Y);
    return mle_corr_core(m, omega);
}

/**
 * The function `lsq_corr_poly` calculates the least squares correlation polynomial for a given set
 * of data.
 *
 * @param[in] Y The parameter `Y` is an input array of type `Arr` which represents the biased
 * covariance matrix.
 * @param site The "site" parameter is a 1-dimensional array or vector that represents the spatial
 * locations of the data points. It is used to construct a polynomial matrix called "Sig" in the
 * lsq_corr_poly function. The size of the "site" array should be the same as the size of the
 * @param m The parameter `m` represents the degree of the polynomial used in the least squares
 * correlation calculation. It determines the number of coefficients in the polynomial.
 *
 * @return The function `lsq_corr_poly` returns a tuple containing three elements:
 * 1. `Arr`: The best solution `omega.x_best()`.
 * 2. `size_t`: The number of iterations `num_iters`.
 */
std::tuple<Arr, size_t> lsq_corr_poly(const Arr &Y, const Arr &site, size_t m) {
    auto Sig = construct_poly_matrix(site, m);
    // omega = mtx_norm_oracle(Sig, Y, a)
    auto a = xt::zeros<double>({m});
    auto Q = QmiOracle<Arr>(Sig, Y);
    auto ellip = Ell<Arr>(10.0, a);
    auto omega = BSearchAdaptor<decltype(Q), decltype(ellip)>(Q, ellip);
    auto [upper, num_iters] = bsearch(omega, std::make_pair(0.0, 100.0 * 100.0));
    return {omega.x_best(), num_iters};
}
