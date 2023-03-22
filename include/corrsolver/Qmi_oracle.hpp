// -*- coding: utf-8 -*-
#pragma once

#include <gsl/span>
#include <lmisolver/ldlt_ext.hpp>
#include <optional>
// #include <xtensor/xarray.hpp>

/*!
 * @brief Oracle for Quadratic Matrix Inequality
 *
 *    This oracle solves the following feasibility problem:
 *
 *        find  x
 *        s.t.  t * I - F(x)' F(x) >= 0
 *
 *    where
 *
 *        F(x) = F0 - (F1 * x1 + F2 * x2 + ...)
 */
template <typename Arr036> class Qmi_oracle {
    using Cut = std::pair<Arr036, double>;

  private:
    double _t = 0.;
    size_t _nx = 0;
    size_t _count = 0;

    const size_t _n;

  public:
    const size_t _m;  // need better sol'n

  private:
    const gsl::span<const Arr036> _F;
    const Arr036 _F0;
    Arr036 _Fx;

  public:
    ldlt_ext<Arr036> _Q;

    /*!
     * @brief Construct a new qmi oracle object
     *
     * @param[in] F
     * @param[in] F0
     */
    Qmi_oracle(gsl::span<const Arr036> F, Arr036 F0);

    /*!
     * @brief Update t
     *
     * @param[in] t the best-so-far optimal value
     */
    auto update(double t) -> void { this->_t = t; }

    /*!
     * @brief
     *
     * @param[in] x
     * @return std::optional<Cut>
     */
    auto assess_feas(const Arr036& x) -> std::optional<Cut>;

    /*!
     * @brief
     *
     * @param[in] x
     * @return std::optional<Cut>
     */
    auto operator()(const Arr036& x) -> std::optional<Cut> { return this->assess_feas(x); }
};
