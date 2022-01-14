#include <stddef.h>  // for size_t

#include <cassert>                      // for assert
#include <corrsolver/Qmi_oracle.hpp>    // for Qmi_oracle, Qmi_oracle::Arr
#include <ellalgo/utility.hpp>          // for zeros
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

using Arr = xt::xarray<double, xt::layout_type::row_major>;
using Cut = std::tuple<Arr, double>;

/*!
 * @brief
 *
 * @param[in] x
 * @return std::optional<Cut>
 */
std::optional<Cut> Qmi_oracle::operator()(const Arr& x) {
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
    Arr g = xt::zeros<double>({this->_nx});
    for (auto k = 0U; k != this->_nx; ++k) {
        const auto Fkp = xt::view(this->_F[k], xt::range(start, stop), xt::all());
        g(k) = -2 * dot(dot(v, Fkp), Av)();
    }
    return {{std::move(g), ep}};
}
