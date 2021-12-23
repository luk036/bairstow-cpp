#include <bairstow/ThreadPool.h>

#include <bairstow/autocorr.hpp>
#include <bairstow/matrix2.hpp>
#include <bairstow/rootfinding.hpp>
#include <bairstow/vector2.hpp>
#include <cmath>  // import pow, cos, sqrt

/**
 * @brief
 *
 * @param[in] pa
 * @return std::vector<vec2>
 */
auto initial_autocorr(const std::vector<double>& pa) -> std::vector<vec2> {
    static const auto PI = std::acos(-1.);

    auto N = pa.size() - 1;
    auto re = std::pow(pa.back(), 1.0 / N);

    N /= 2;
    auto k = PI / N;
    auto m = re * re;
    auto vr0s = std::vector<vec2>{};
    for (auto i = 1U; i < N; i += 2) {
        vr0s.emplace_back(vec2{2 * re * std::cos(k * i), m});
    }
    return vr0s;
}

/**
 * @brief Multi-threading Bairstow's method (even degree only)
 *
 * @param[in] pa polynomial
 * @param[in,out] vrs vector of iterates
 * @param[in] options maximum iterations and tolorance
 * @return std::tuple<unsigned int, bool>
 */
auto pbairstow_autocorr(const std::vector<double>& pa, std::vector<vec2>& vrs,
                        const Options& options = Options()) -> std::tuple<unsigned int, bool> {
    auto N = pa.size() - 1;  // degree, assume even
    auto M = vrs.size();
    auto found = false;
    auto converged = std::vector<bool>(M, false);
    auto niter = 1U;
    ThreadPool pool(std::thread::hardware_concurrency());

    for (; niter != options.max_iter; ++niter) {
        auto tol = 0.0;
        std::vector<std::future<double>> results;

        for (auto i = 0U; i != M && !converged[i]; ++i) {
            results.emplace_back(pool.enqueue([&, i]() {
                auto pb = pa;
                // auto n = pa.size() - 1;
                const auto& vri = vrs[i];
                const auto vA = horner(pb, N, vri);
                const auto tol_i = vA.norm_inf();
                if (tol_i < options.tol) {
                    converged[i] = true;
                    return tol_i;
                }
                // tol = std::max(tol, toli);
                auto vA1 = horner(pb, N - 2, vri);
                auto vrin = numeric::vector2<double>(-vri.x(), 1.0) / vri.y();
                vA1 -= delta(vA, vrin, vri - vrin);

                for (auto j = 0U; j != M && j != i; ++j) {  // exclude i
                    auto vrj = vrs[j];                      // make a copy, don't reference!
                    vA1 -= delta(vA, vrj, vri - vrj);
                    auto vrjn = numeric::vector2<double>(-vrj.x(), 1.0) / vrj.y();
                    vA1 -= delta(vA, vrjn, vri - vrjn);
                }
                vrs[i] -= delta(vA, vri, std::move(vA1));  // Gauss-Seidel fashion
                return tol_i;
            }));
        }
        for (auto&& result : results) {
            tol = std::max(tol, result.get());
        }
        // fmt::print("tol: {}\n", tol);
        if (tol < options.tol) {
            found = true;
            break;
        }
    }
    return {niter, found};
}

// auto find_rootq(const vec2& r) {
//     auto hb = b / 2.;
//     auto d = hb * hb - c;
//     if (d < 0.) {
//         auto x1 = -hb + (sqrt(-d) if (hb < 0. else -sqrt(-d))*1j;
//     }
//     else {
//         auto x1 = -hb + (sqrt(d) if (hb < 0. else -sqrt(d));
//     }
//     auto x2 = c / x1;
//     return x1, x2;
// }
