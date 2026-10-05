#include <corrsolver/layouts.hpp>
#include <corrsolver/linalg.hpp>
#include <cstddef>
#include <utility>

InitialGuess lsq_initial_guess(const Arr& Y, size_t m, const SolverConfig& cfg) {
    const auto normY = cfg.lsq_norm_scale * norm(Y);
    const auto normY2 = cfg.lsq_frob_scale * normY * normY;
    std::valarray<double> radii(cfg.lsq_aug_r0, m + 1);
    radii[m] = normY2 * normY2;
    Arr x = zeros(m + 1);
    x[0] = 4;
    x[m] = normY2 / 2.0;
    return {std::move(x), std::move(radii), 0.0};
}

InitialGuess mle_initial_guess(size_t m, const SolverConfig& cfg) {
    Arr x = zeros(m);
    x[0] = 4.0;
    return {std::move(x), {}, cfg.mle_r0};
}

InitialGuess cccp_initial_guess(const Arr& x, const SolverConfig& cfg) {
    return {x, {}, cfg.mle_r0};
}

Ell<Arr> make_ellipsoid(InitialGuess& guess) {
    if (guess.radii.size() != 0) return Ell<Arr>(guess.radii, std::move(guess.x));
    return Ell<Arr>(guess.alpha, std::move(guess.x));
}
