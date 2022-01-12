#include <bairstow/ThreadPool.h>  // for ThreadPool

#include <__bit_reference>           // for __bit_reference
#include <algorithm>                 // for max
#include <bairstow/autocorr.hpp>     // for extract_autocorr, initial_autocorr
#include <bairstow/rootfinding.hpp>  // for vec2, delta, horner, Options
#include <bairstow/vector2.hpp>      // for vector2, operator-, operator/
#include <cmath>                     // for abs, sqrt, acos, cos, pow
#include <functional>                // for __base
#include <future>                    // for future
#include <thread>                    // for thread
#include <tuple>                     // for tuple
#include <type_traits>               // for move
#include <vector>                    // for vector, vector<>::reference, __v...

/**
 * @brief initial guess (specific for auto-correlation function)
 *
 * @param[in] pa
 * @return std::vector<vec2>
 */
auto initial_autocorr(const std::vector<double>& pa) -> std::vector<vec2> {
    static const auto PI = std::acos(-1.);

    auto N = int(pa.size()) - 1;
    const auto re = std::pow(std::abs(pa[N]), 1.0 / N);

    N /= 2;
    const auto k = PI / N;
    const auto m = re * re;
    auto vr0s = std::vector<vec2>{};
    for (auto i = 1; i < N; i += 2) {
        vr0s.emplace_back(vec2{-2 * re * std::cos(k * i), m});
    }
    return vr0s;
}

/**
 * @brief Multi-threading Bairstow's method (specific for auto-correlation function)
 *
 * @param[in] pa polynomial
 * @param[in,out] vrs vector of iterates
 * @param[in] options maximum iterations and tolorance
 * @return std::tuple<unsigned int, bool>
 */
auto pbairstow_autocorr(const std::vector<double>& pa, std::vector<vec2>& vrs,
                        const Options& options = Options()) -> std::tuple<unsigned int, bool> {
    const auto N = pa.size() - 1;  // degree, assume even
    const auto M = vrs.size();
    auto found = false;
    auto converged = std::vector<bool>(M, false);
    auto niter = 1U;
    ThreadPool pool(std::thread::hardware_concurrency());

    for (; niter != options.max_iter; ++niter) {
        auto tol = 0.0;
        std::vector<std::future<double>> results;
        for (auto i = 0U; i != M; ++i) {
            if (converged[i]) {
                continue;
            }
            results.emplace_back(pool.enqueue([&, i]() {
                auto pb = pa;
                const auto& vri = vrs[i];
                const auto vA = horner(pb, N, vri);
                const auto tol_i = vA.norm_inf();
                if (tol_i < 1e-15) {
                    converged[i] = true;
                    return tol_i;
                }
                auto vA1 = horner(pb, N - 2, vri);
                for (auto j = 0U; j != M; ++j) {  // exclude i
                    if (j == i) {
                        continue;
                    }
                    const auto vrj = vrs[j];  // make a copy, don't reference!
                    vA1 -= delta(vA, vrj, vri - vrj);
                    const auto vrjn = numeric::vector2<double>(vrj.x(), 1.0) / vrj.y();
                    vA1 -= delta(vA, vrjn, vri - vrjn);
                }
                const auto vrin = numeric::vector2<double>(vri.x(), 1.0) / vri.y();
                vA1 -= delta(vA, vrin, vri - vrin);

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

/**
 * @brief Extract the quadratic function where its roots are within a unit circle
 *
 *   x^2 + r*x + t or x^2 + (r/t) * x + (1/t)
 *   (x + a1)(x + a2) = x^2 + (a1 + a2) x + a1 * a2
 *
 * @param[in,out] vr
 */
void extract_autocorr(vec2& vr) {
    const auto& r = vr.x();
    const auto& t = vr.y();
    const auto hr = r / 2.0;
    const auto d = hr * hr - t;
    if (d < 0.0) {  // complex conjugate root
        if (t > 1.0) {
            vr = vec2{r, 1.0} / t;
        }
        // else no need to change
    } else {  // two real roots
        auto a1 = hr + (hr >= 0.0 ? sqrt(d) : -sqrt(d));
        auto a2 = t / a1;
        if (std::abs(a1) > 1.0) {
            if (std::abs(a2) > 1.0) {
                a2 = 1.0 / a2;
            }
            a1 = 1.0 / a1;
            vr = vec2{a1 + a2, a1 * a2};
        } else if (std::abs(a2) > 1.0) {
            a2 = 1.0 / a2;
            vr = vec2{a1 + a2, a1 * a2};
        }
        // else no need to change
    }
}
