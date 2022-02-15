#include <stddef.h>  // for size_t

#include <cassert>                      // for assert
#include <corrsolver/Qmi_oracle.hpp>    // for Qmi_oracle, Qmi_oracle::Arr
#include <gsl/span>                     // for span, span<>::reference
#include <lmisolver/ldlt_ext.hpp>       // for ldlt_ext
#include <optional>                     // for optional
#include <tuple>                        // for tuple
#include <type_traits>                  // for move
#include <xtensor-blas/xlinalg.hpp>     // for dot
#include <xtensor/xarray.hpp>           // for xarray_container
#include <xtensor/xcontainer.hpp>       // for xcontainer<>::const_reference
#include <xtensor/xlayout.hpp>          // for layout_type, layout_type::...
#include <xtensor/xoperation.hpp>       // for operator*, xfunction_type_t
#include <xtensor/xsemantic.hpp>        // for xsemantic_base
#include <xtensor/xslice.hpp>           // for range, all
#include <xtensor/xtensor_forward.hpp>  // for xarray
#include <xtensor/xview.hpp>            // for xview, row, view, col

// #define ROW(X, index) xt::view(X, index, xt::all())
// #define COLUMN(X, index) xt::view(X, xt::all(), index)

/*!
 * @brief Construct a new qmi oracle object
 *
 * @param[in] F
 * @param[in] F0
 */
template <typename Arr036> Qmi_oracle<Arr036>::Qmi_oracle(gsl::span<const Arr036> F, Arr036 F0)
    : _n{F0.shape()[0]},
      _m{F0.shape()[1]},
      _F{F},
      _F0{std::move(F0)},
      _Fx{xt::zeros<double>({_m, _n})},  // transposed
      _Q(_m)                             // take column
{}

/*!
 * @brief
 *
 * @param[in] x
 * @return std::optional<Cut>
 */
template <typename Arr036> auto Qmi_oracle<Arr036>::operator()(const Arr036& x)
    -> std::optional<typename Qmi_oracle<Arr036>::Cut> {
    using xt::linalg::dot;

    this->_count = 0;
    this->_nx = x.shape()[0];

    auto getA = [&, this](size_t i, size_t j) -> double {  // ???
        assert(i >= j);
        if (this->_count < i + 1) {
            this->_count = i + 1;
            xt::row(this->_Fx, i) = xt::col(this->_F0, i);
            for (auto k = 0U; k != this->_nx; ++k) {
                xt::row(this->_Fx, i) -= xt::col(this->_F[k], i) * x(k);
            }
        }
        auto a = -dot(xt::row(this->_Fx, i), xt::row(this->_Fx, j))();
        if (i == j) {
            a += this->_t;
        }
        return a;
    };

    if (this->_Q.factor(getA)) {
        return {};
    }

    const auto ep = this->_Q.witness();
    const auto [start, stop] = this->_Q.p;
    const auto v = xt::view(this->_Q.witness_vec, xt::range(start, stop), xt::all());
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
template class Qmi_oracle<Arr>;