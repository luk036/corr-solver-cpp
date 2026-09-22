/** @file oracles.hpp
 *  @brief Separation oracles for the LSQ, MLE and CCP correlation fits.
 */

#pragma once

#include <corrsolver/linalg.hpp>
#include <corrsolver/qmi_oracle.hpp>
#include <corrsolver/types.hpp>
#include <cstddef>
#include <ellalgo/arr.hpp>
#include <ellalgo/oracles/lmi0_oracle.hpp>
#include <ellalgo/oracles/lmi_oracle.hpp>
#include <tuple>
#include <utility>
#include <vector>

/// Cholesky-based scratch shared by the MLE and CCP oracles: the factor R with
/// Omega(x) = R^T R, S = R^-T R^-1, and SY = S Y.
struct MleScratch {
    Arr R;
    Arr invR;
    Arr S;
    Arr SY;

    explicit MleScratch(size_t n) : R(n, n), invR(n, n), S(n, n), SY(n, n) {}

    void update(Lmi0Oracle<Arr>& lmi0, const Arr& Y) {
        lmi0._mq.sqrt(R);
        invR = inv_upper_tri(R);
        S = matmul(invR, transpose(invR));
        SY = matmul(S, Y);
    }
};

/// Fold the best-so-far value into a cut, reporting whether t improved.
inline std::tuple<Cut, bool> optim_cut(Arr g, double value, double& t) {
    auto f = value - t;
    auto shrunk = false;
    if (f < 0.0) {
        t = value;
        f = 0.0;
        shrunk = true;
    }
    return {{std::move(g), f}, shrunk};
}

/// Least-squares oracle for min ||F0 - F(x)|| s.t. F(x) >= 0, with the trailing
/// variable x(n-1) acting as the objective.
class LsqOracle {
    QmiOracle<Arr> _qmi;
    Lmi0Oracle<Arr> _lmi0;

  public:
    LsqOracle(size_t m, const std::vector<Arr>& F, const Arr& F0) : _qmi(F, F0), _lmi0(m, F) {}
    std::tuple<Cut, bool> assess_optim(const Arr& x, double& t);
};

/// Maximum-likelihood oracle for min log det Omega + Tr(Omega^-1 Y) s.t.
/// 2Y >= Omega >= 0.
class MleOracle {
    Arr Y_;
    std::vector<Arr> sig_;
    Lmi0Oracle<Arr> _lmi0;
    LmiOracle<Arr> _lmi;
    MleScratch _scratch;
    Arr _V;

  public:
    MleOracle(size_t m, const std::vector<Arr>& Sig, const Arr& Y);
    std::tuple<Cut, bool> assess_optim(const Arr& x, double& t);
};

/// One CCP round of the MLE surrogate, linearized at M = Omega_k^-1.
class CccpMleOracle {
    Arr Y_;
    std::vector<Arr> sig_;
    Lmi0Oracle<Arr> _lmi0;
    Arr _mk;
    MleScratch _scratch;
    Arr _SYS;

  public:
    CccpMleOracle(size_t ndim, const std::vector<Arr>& Sig, const Arr& Y, const Arr& M);
    std::tuple<Cut, bool> assess_optim(const Arr& x, double& t);
};
