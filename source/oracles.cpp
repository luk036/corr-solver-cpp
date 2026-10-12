#include <cmath>
#include <corrsolver/linalg.hpp>
#include <corrsolver/oracles.hpp>
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
        for (size_t i = 0; i < n - 1; ++i) g(i) = g1(i);
        g(n - 1) = -this->_qmi.witness_norm_sq();
        return {{std::move(g), f1}, false};
    }
    g(n - 1) = 1.0;
    if (auto f0 = x(n - 1) - t; f0 > 0) {
        return {{std::move(g), f0}, false};
    }
    t = x(n - 1);
    return {{std::move(g), 0.0}, true};
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
