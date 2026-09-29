#include <cmath>
#include <corrsolver/geometry.hpp>
#include <corrsolver/kernels.hpp>
#include <corrsolver/linalg.hpp>
#include <corrsolver/sites.hpp>
#include <cstddef>
#include <random>

Arr create_2d_sites(size_t nx, size_t ny) {
    auto sx = linspace(0.0, 10.0, nx);
    auto sy = linspace(0.0, 8.0, ny);
    auto [xx, yy] = meshgrid(sx, sy);
    auto st = stack(flatten(xx), flatten(yy));
    return transpose(st);
}

Arr sample_covariance(const Arr& Sigma, size_t N, std::mt19937_64* rng, double var, double tau) {
    auto n = Sigma.rows();
    std::mt19937_64 local_rng(5);
    std::mt19937_64& gen = (rng != nullptr) ? *rng : local_rng;

    auto A = cholesky(Sigma);
    Arr Y(n, n);
    for (size_t k = 0; k < N; ++k) {
        auto x = var * randn(n, gen);
        Arr y(n);
        for (size_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (size_t j = 0; j < n; ++j) s += A(i, j) * x(j);
            y(i) = s;
        }
        auto noise = randn(n, gen);
        for (size_t i = 0; i < n; ++i) y(i) += tau * noise(i);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) Y(i, j) += y(i) * y(j);
    }
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) Y(i, j) /= static_cast<double>(N);
    return Y;
}

Arr create_2d_isotropic(const Arr& site, size_t N, std::mt19937_64* rng) {
    const double sdkern = 0.3;
    auto D = construct_distance_matrix(site);
    auto n = D.rows();
    Arr Sig(n, n);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) Sig(i, j) = exponential_kernel(D(i, j), sdkern);
    return sample_covariance(Sig, N, rng);
}
