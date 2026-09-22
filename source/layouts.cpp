#include <corrsolver/layouts.hpp>
#include <corrsolver/linalg.hpp>
#include <cstddef>
#include <utility>

InitialGuess lsq_initial_guess(const Arr& Y, size_t m) {
    const auto normY = 100.0 * norm(Y);
    const auto normY2 = 32.0 * normY * normY;
    std::valarray<double> radii(256.0, m + 1);
    radii[m] = normY2 * normY2;
    Arr x = zeros(m + 1);
    x[0] = 4;
    x[m] = normY2 / 2.0;
    return {std::move(x), std::move(radii), 0.0};
}

InitialGuess mle_initial_guess(size_t m) {
    Arr x = zeros(m);
    x[0] = 4.0;
    return {std::move(x), {}, 500.0};
}

InitialGuess cccp_initial_guess(const Arr& x) { return {x, {}, 100.0}; }

Ell<Arr> make_ellipsoid(InitialGuess& guess) {
    if (guess.radii.size() != 0) return Ell<Arr>(guess.radii, std::move(guess.x));
    return Ell<Arr>(guess.alpha, std::move(guess.x));
}
