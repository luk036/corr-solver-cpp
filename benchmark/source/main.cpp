#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include <cstddef>
#include <cstdint>
#include <ellalgo/arr.hpp>
#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>

// Implementation lives in source/lsq_corr_ell.cpp; this benchmark only drives it
// so that fixes and optimizations cannot drift between the two copies.
extern Arr create_2d_sites(size_t, size_t);
extern Arr create_2d_isotropic(const Arr&, size_t);
extern std::vector<Arr> construct_poly_matrix(const Arr&, size_t);
extern std::tuple<Arr, size_t> lsq_corr_poly2(const Arr&, const Arr&, size_t);
extern std::tuple<Arr, size_t> mle_corr_poly(const Arr&, const Arr&, size_t);

int main() {
    constexpr size_t nx = 10;
    constexpr size_t ny = 8;
    constexpr size_t m = 4;

    // Generate data
    std::cout << "Generating data...\n";
    auto site = create_2d_sites(nx, ny);
    auto Y = create_2d_isotropic(site, 3000);
    auto Sig = construct_poly_matrix(site, m);
    auto n_sites = Y.rows();
    (void)n_sites;

    // Export Y matrix to binary file for Rust benchmark to read
    {
        std::ofstream fout("benchmark_data.bin", std::ios::binary);
        uint64_t n = n_sites;
        fout.write(reinterpret_cast<const char*>(&n), sizeof(n));
        for (size_t i = 0; i < n_sites; ++i)
            for (size_t j = 0; j < n_sites; ++j) {
                double val = Y(i, j);
                fout.write(reinterpret_cast<const char*>(&val), sizeof(val));
            }
        // Write site matrix
        uint64_t ns = site.rows();
        uint64_t nd = site.cols();
        fout.write(reinterpret_cast<const char*>(&ns), sizeof(ns));
        fout.write(reinterpret_cast<const char*>(&nd), sizeof(nd));
        for (size_t i = 0; i < ns; ++i)
            for (size_t j = 0; j < nd; ++j) {
                double val = site(i, j);
                fout.write(reinterpret_cast<const char*>(&val), sizeof(val));
            }
        // Write poly matrix Sig
        uint64_t nk = Sig.size();
        fout.write(reinterpret_cast<const char*>(&nk), sizeof(nk));
        for (size_t k = 0; k < nk; ++k) {
            for (size_t i = 0; i < ns; ++i)
                for (size_t j = 0; j < ns; ++j) {
                    double val = Sig[k](i, j);
                    fout.write(reinterpret_cast<const char*>(&val), sizeof(val));
                }
        }
        fout.close();
        std::cout << "Data exported to benchmark_data.bin\n";
    }

    // --- LSQ benchmark ---
    std::cout << "\n=== LSQ Correlation ===\n";
    {
        ankerl::nanobench::Bench bench;
        bench.title("LSQ correlation").unit("op").warmup(1).epochs(5).minEpochIterations(1);

        Arr lsq_coeffs;
        size_t lsq_iters = 0;
        bench.run("LSQ_corr", [&] {
            auto result = lsq_corr_poly2(Y, site, m);
            lsq_coeffs = std::get<0>(result);
            lsq_iters = std::get<1>(result);
            ankerl::nanobench::doNotOptimizeAway(result);
        });
        std::cout << "  coeffs = [";
        for (size_t i = 0; i < m; ++i) std::cout << lsq_coeffs(i) << (i + 1 < m ? ", " : "");
        std::cout << "]\n";
        std::cout << "  iters = " << lsq_iters << "\n";
    }

    // --- MLE benchmark ---
    std::cout << "\n=== MLE Correlation ===\n";
    {
        ankerl::nanobench::Bench bench;
        bench.title("MLE correlation").unit("op").warmup(1).epochs(5).minEpochIterations(1);

        size_t mle_iters = 0;
        bench.run("MLE_corr", [&] {
            auto result = mle_corr_poly(Y, site, m);
            mle_iters = std::get<1>(result);
            ankerl::nanobench::doNotOptimizeAway(result);
        });
        std::cout << "  iters = " << mle_iters << "\n";
    }

    return 0;
}
