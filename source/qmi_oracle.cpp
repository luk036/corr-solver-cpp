#include <cassert>
#include <corrsolver/qmi_oracle.hpp>
#include <cstddef>
#include <ellalgo/arr.hpp>
#include <ellalgo/oracles/ldlt_mgr.hpp>
#include <optional>

template <typename Arr036> QmiOracle<Arr036>::QmiOracle(const std::vector<Arr036>& F, Arr036 F0)
    : _n{F0.rows()}, _m{F0.cols()}, m_F{F}, m_F0{std::move(F0)}, m_Fx{zeros(_m, _n)}, _mq(_m) {}

template <typename Arr036> auto QmiOracle<Arr036>::assess_feas(const Arr036& x)
    -> std::optional<typename QmiOracle<Arr036>::Cut> {
    this->_count = 0;
    this->_nx = x.size();

    auto getA = [&x, this](size_t i, size_t j) -> double {
        assert(i >= j);
        if (this->_count < i + 1) {
            this->_count = i + 1;
            for (size_t c = 0; c < this->_m; ++c) this->m_Fx(i, c) = this->m_F0(c, i);
            for (size_t k = 0; k != this->_nx; ++k)
                for (size_t c = 0; c < this->_m; ++c) this->m_Fx(i, c) -= this->m_F[k](c, i) * x(k);
        }
        double a = 0.0;
        for (size_t c = 0; c < this->_m; ++c) a -= this->m_Fx(i, c) * this->m_Fx(j, c);
        if (i == j) a += this->_t;
        return a;
    };

    if (this->_mq.factor(getA)) return {};

    const auto ep = this->_mq.witness();
    const auto [start, stop] = this->_mq.pos;
    Arr036 wit_vec = zeros(this->_m);
    this->_mq.set_witness_vec(wit_vec);

    auto v_len = stop - start;
    Arr036 v(v_len);
    for (size_t c = 0; c < v_len; ++c) v(c) = wit_vec(start + c);

    auto ncols = this->m_Fx.cols();
    Arr036 Fxp(stop - start, ncols);
    for (size_t r = 0; r < stop - start; ++r)
        for (size_t c = 0; c < ncols; ++c) Fxp(r, c) = this->m_Fx(start + r, c);

    Arr036 Av(ncols);
    for (size_t c = 0; c < ncols; ++c) {
        double s = 0.0;
        for (size_t r = 0; r < v_len; ++r) s += v(r) * Fxp(r, c);
        Av(c) = s;
    }

    Arr036 g = zeros(this->_nx);
    for (size_t k = 0; k != this->_nx; ++k) {
        const auto& Fk = this->m_F[k];
        double vFkp = 0.0;
        for (size_t r = 0; r < v_len; ++r) {
            double row_dot = 0.0;
            for (size_t c = 0; c < ncols; ++c) row_dot += Fk(start + r, c) * Av(c);
            vFkp += v(r) * row_dot;
        }
        g(k) = -2.0 * vFkp;
    }
    return {{std::move(g), ep}};
}

template class QmiOracle<Arr>;
