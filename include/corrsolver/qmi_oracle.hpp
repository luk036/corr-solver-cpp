// -*- coding: utf-8 -*-
#pragma once

// Disable svector on macOS to avoid Clang template ambiguity issues
// where long and unsigned long are both 64-bit
#ifdef __APPLE__
#    define XTENSOR_DISABLE_SVECTOR 1
#endif

#include <ellalgo/oracles/ldlt_mgr.hpp>
#include <optional>
#include <vector>

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
template <typename Arr036> class QmiOracle {
    using Cut = std::pair<Arr036, double>;

  private:
    double _t = 0.;     //!< Current best-so-far optimal value (t parameter)
    size_t _nx = 0;     //!< Number of variables (size of x)
    size_t _count = 0;  //!< Counter for caching evaluated matrices

    const size_t _n;  //!< Number of rows (matrix dimension)

  public:
    const size_t _m;  //!< Number of columns (matrix dimension)

  private:
    const std::vector<Arr036>& _F;  //!< Vector of coefficient matrices F_k
    const Arr036 _F0;               //!< Base matrix F0
    Arr036 _Fx;                     //!< Evaluated matrix F(x) = F0 - sum(F_k * x_k)

  public:
    LDLTMgr _mq;  //!< LDLT matrix manager for factorization

    /**
     * @brief Construct a new QmiOracle object
     * @param[in] F Vector of coefficient matrices F_k for k = 1, 2, ...
     * @param[in] F0 Base matrix F0 in the quadratic matrix inequality
     */
    QmiOracle(const std::vector<Arr036>& F, Arr036 F0);

    /**
     * @brief Update the best-so-far optimal value
     * @param[in] t The current best feasible objective value
     */
    auto update(double t) -> void { this->_t = t; }

    /*!
     * @brief Assess feasibility and generate cutting plane
     * @param[in] x Current point to evaluate
     * @return Optional cut (gradient and violation) if infeasible, nullopt if feasible
     */
    auto assess_feas(const Arr036& x) -> std::optional<Cut>;
};
