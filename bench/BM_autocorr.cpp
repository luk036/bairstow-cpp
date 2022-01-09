#include <bairstow/autocorr.hpp>     // import pbairstow_autocorr, initial_autocorr
#include <bairstow/rootfinding.hpp>  // import horner

#include "benchmark/benchmark.h"

auto run_autocorr() {
    auto r = std::vector<double>{10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0};
    auto vrs = initial_autocorr(r);
    auto options = Options();
    options.tol = 1e-12;
    auto result = pbairstow_autocorr(r, vrs, options);
    return result;
}

auto run_pbairstow() {
    auto r = std::vector<double>{10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0};
    auto vrs = initial_guess(r);
    auto options = Options();
    options.tol = 1e-12;
    auto result = pbairstow_even(r, vrs, options);
    return result;
}

/**
 * @brief
 *
 * @param[in,out] state
 */
static void Autocorr(benchmark::State& state) {
    while (state.KeepRunning()) {
        run_autocorr();
    }
}

// Register the function as a benchmark
BENCHMARK(Autocorr);

/**
 * @brief
 *
 * @param[in,out] state
 */
static void PBairstow(benchmark::State& state) {
    while (state.KeepRunning()) {
        run_pbairstow();
    }
}

// Register the function as a benchmark
BENCHMARK(PBairstow);

BENCHMARK_MAIN();