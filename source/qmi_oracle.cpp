#include <stddef.h>  // for size_t

#include <cassert>                       // for assert
#include <corrsolver/qmi_oracle.hpp>     // for QmiOracle, QmiOracle::Arr
#include <ellalgo/oracles/ldlt_mgr.hpp>  // for LDLTMgr
#include <optional>                      // for optional
#include <tuple>                         // for tuple
#include <type_traits>                   // for move
#include <xtensor-blas/xlinalg.hpp>      // for dot
#include <xtensor/xarray.hpp>            // for xarray_container
#include <xtensor/xcontainer.hpp>        // for xcontainer<>::const_reference
#include <xtensor/xlayout.hpp>           // for layout_type, layout_type::...
#include <xtensor/xoperation.hpp>        // for operator*, xfunction_type_t
#include <xtensor/xsemantic.hpp>         // for xsemantic_base
#include <xtensor/xslice.hpp>            // for range, all
#include <xtensor/xtensor_forward.hpp>   // for xarray
#include <xtensor/xview.hpp>             // for xview, row, view, col

/**
 * @brief Construct a new quadratic matrix inequality oracle object
 *
 * The code snippet is defining the constructor for the `QmiOracle` class template. It takes two
 * parameters, `F` and `F0`, which are vectors of `Arr036` objects.
 *
 * @param[in] F
 * @param[in] F0
 */
template <typename Arr036> QmiOracle<Arr036>::QmiOracle(const std::vector<Arr036> &F, Arr036 F0)
    : _n{F0.shape()[0]},
      _m{F0.shape()[1]},
      _F{F},
      _F0{std::move(F0)},
      _Fx{xt::zeros<double>({_m, _n})},  // transposed
      _mq(_m)                            // take column
{}

/* The code snippet is defining a member function `assess_feas` for the `QmiOracle` class template.
This function takes a parameter `x` of type `Arr036` and returns an `std::optional` object
containing a `Cut` object. */
template <typename Arr036> auto QmiOracle<Arr036>::assess_feas(const Arr036 &x)
    -> std::optional<typename QmiOracle<Arr036>::Cut> {
    using xt::linalg::dot;

    this->_count = 0;
    this->_nx = x.shape()[0];

    auto getA = [&x, this](size_t i, size_t j) -> double {  // ???
        assert(i >= j);
        auto ii = int(i);
        auto ij = int(j);
        if (this->_count < i + 1) {
            this->_count = i + 1;
            xt::row(this->_Fx, ii) = xt::col(this->_F0, ii);
            for (auto k = 0U; k != this->_nx; ++k) {
                xt::row(this->_Fx, ii) -= xt::col(this->_F[k], ii) * x(k);
            }
        }
        auto a = -dot(xt::row(this->_Fx, ii), xt::row(this->_Fx, ij))();
        if (i == j) {
            a += this->_t;
        }
        return a;
    };

    if (this->_mq.factor(getA)) {
        return {};
    }

    const auto ep = this->_mq.witness();
    const auto [start, stop] = this->_mq.pos;
    Arr036 wit_vec = xt::zeros<double>({this->_m});
    this->_mq.set_witness_vec(wit_vec);
    const auto v = xt::view(wit_vec, xt::range(start, stop), xt::all());
    const auto Fxp = xt::view(this->_Fx, xt::range(start, stop));
    const auto Av = dot(v, Fxp);
    Arr036 g = xt::zeros<double>({this->_nx});
    for (auto k = 0U; k != this->_nx; ++k) {
        const auto Fkp = xt::view(this->_F[k], xt::range(start, stop), xt::all());
        g(k) = -2.0 * dot(dot(v, Fkp), Av)();
    }
    return {{std::move(g), ep}};
}

using Arr = xt::xarray<double, xt::layout_type::row_major>;
template class QmiOracle<Arr>;
