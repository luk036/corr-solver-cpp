#include <corrsolver/linalg.hpp>
#include <corrsolver/oracles.hpp>
#include <cmath>
#include <cstddef>
#include <ellalgo/oracles/ldlt_mgr.hpp>
#include <tuple>
#include <utility>
#include <vector>

std::tuple<Cut, bool> LsqOracle::assess_optim(const Arr& x, double& t) {
    auto n = x.size();
    Arr g = zeros(n);

    // v = x[0..n-2]
    Arr v(n - 1);
    for (size_t i = 0; i < n - 1; ++i) v(i) = x(i);

    if (auto* cut0 = this->_lmi0.assess_feas(v)) {
        const auto& [g0, f0] = *cut0;
        for (size_t i = 0; i < n - 1; ++i) g(i) = g0(i);
        g(n - 1) = 0.0;
        return {{std::move(g), f0}, false};
    }
    this->_qmi.update(x(n - 1));

    if (auto cut1 = this->_qmi.assess_feas(v)) {
        const auto& [g1, f1] = *cut1;
        const auto& Q = this->_qmi._mq;
        const auto& [start, stop] = Q.pos;
        Arr wit_vec = zeros(this->_qmi._m);
        Q.set_witness_vec(wit_vec);

        double v2norm2 = 0.0;
        for (size_t i = start; i < stop; ++i) v2norm2 += wit_vec(i) * wit_vec(i);

        for (size_t i = 0; i < n - 1; ++i) g(i) = g1(i);
        g(n - 1) = -v2norm2;
        return {{std::move(g), f1}, false};
    }
    g(n - 1) = 1.0;
    if (auto f0 = x(n - 1) - t; f0 > 0) {
        return {{std::move(g), f0}, false};
    }
    t = x(n - 1);
    return {{std::move(g), 0.0}, true};
}

MleOracle::MleOracle(size_t m, const std::vector<Arr>& Sig, const Arr& Y)
    : Y_{Y},
      sig_{Sig},
      _lmi0(m, Sig),
      _lmi(m, Sig, 2.0 * Y),
      _scratch(Y.rows()),
      _V(Y.rows(), Y.rows()) {}

std::tuple<Cut, bool> MleOracle::assess_optim(const Arr& x, double& t) {
    if (auto* cut1 = this->_lmi.assess_feas(x)) return {*cut1, false};
    if (auto* cut0 = this->_lmi0.assess_feas(x)) return {*cut0, false};

    auto n = x.size();
    auto dim = this->_lmi0._mq._n;

    this->_scratch.update(this->_lmi0, this->Y_);
    const auto& S = this->_scratch.S;
    const auto& SY = this->_scratch.SY;

    double log_sum = 0.0;
    for (size_t i = 0; i < dim; ++i) log_sum += std::log(this->_scratch.R(i, i));
    auto f1 = 2.0 * log_sum + trace(SY);

    // g[i] = tr(S Sigma_i) - tr(Sigma_i S Y S) = <(S - SY S)^T, Sigma_i>
    //      = <S - SY S, Sigma_i>   (Sigma_i is symmetric)
    for (size_t a = 0; a < dim; ++a)
        for (size_t b = 0; b < dim; ++b) {
            double s = 0.0;
            for (size_t k = 0; k < dim; ++k) s += SY(a, k) * S(k, b);
            this->_V(a, b) = S(a, b) - s;
        }

    Arr g = zeros(n);
    for (size_t i = 0; i < n; ++i) g(i) = frob_inner(this->_V, this->sig_[i]);
    return optim_cut(std::move(g), f1, t);
}

CccpMleOracle::CccpMleOracle(size_t ndim, const std::vector<Arr>& Sig, const Arr& Y, const Arr& M)
    : Y_{Y},
      sig_{Sig},
      _lmi0(ndim, Sig),
      _mk(zeros(Sig.size())),
      _scratch(Y.rows()),
      _SYS(Y.rows(), Y.rows()) {
    for (size_t i = 0; i < Sig.size(); ++i) this->_mk(i) = trace(matmul(M, Sig[i]));
}

std::tuple<Cut, bool> CccpMleOracle::assess_optim(const Arr& x, double& t) {
    if (auto* cut = this->_lmi0.assess_feas(x)) return {*cut, false};

    auto n = x.size();

    this->_scratch.update(this->_lmi0, this->Y_);
    this->_SYS = matmul(this->_scratch.SY, this->_scratch.S);

    double h = trace(this->_scratch.SY);
    for (size_t i = 0; i < n; ++i) h += x(i) * this->_mk(i);

    Arr g = zeros(n);
    for (size_t i = 0; i < n; ++i) g(i) = -frob_inner(this->sig_[i], this->_SYS) + this->_mk(i);
    return optim_cut(std::move(g), h, t);
}
