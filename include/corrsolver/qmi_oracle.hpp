#pragma once

#include <ellalgo/arr.hpp>
#include <ellalgo/oracles/ldlt_mgr.hpp>
#include <optional>
#include <type_traits>
#include <vector>

template <typename Arr036>
class QmiOracle {
    static_assert(std::is_class_v<Arr036>, "Arr036 must be a class type (e.g. Arr)");
    using Cut = std::pair<Arr036, double>;

  private:
    double _t = 0.;
    size_t _nx = 0;
    size_t _count = 0;
    size_t _n;

  public:
    size_t _m;

  private:
    const std::vector<Arr036>& m_F;
    Arr036 m_F0;
    Arr036 m_Fx;

  public:
    LDLTMgr _mq;

    QmiOracle(const std::vector<Arr036>& F, Arr036 F0);
    auto update(double t) -> void { this->_t = t; }
    auto assess_feas(const Arr036& x) -> std::optional<Cut>;
};
