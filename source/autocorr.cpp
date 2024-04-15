#include <bairstow/ThreadPool.h>  // for ThreadPool

#include <bairstow/robin.hpp>        // for Robin
#include <bairstow/rootfinding.hpp>  // for Vec2, delta, Options, horner_eval
#include <bairstow/vector2.hpp>      // for operator-, Vector2
#include <cmath>                     // for abs, acos, cos, pow
#include <cstddef>                   // for size_t
#include <functional>                // for __base
#include <future>                    // for future
#include <thread>                    // for thread
#include <type_traits>               // for move
#include <utility>                   // for pair
#include <vector>                    // for vector, vector<>::reference, __v...

#ifndef M_PI
#    define M_PI 3.14159265358979323846264338327950288
#endif

/**
 * The function calculates the initial autocorrelation values (specific for
 * auto-correlation function)
 *
 * @param[in] coeffs The parameter `coeffs` is a vector of doubles.
 *
 * @return The function `initial_autocorr` returns a vector of `Vec2` objects.
 */
auto initial_autocorr(const std::vector<double> &coeffs) -> std::vector<Vec2> {
    auto degree = coeffs.size() - 1;
    const auto re = std::pow(std::abs(coeffs[degree]), 1.0 / double(degree));

    degree /= 2;
    const auto k = M_PI / double(degree);
    const auto m = re * re;
    auto vr0s = std::vector<Vec2>{};
    for (auto i = 1U; i < degree; i += 2) {
        vr0s.emplace_back(Vec2{2 * re * std::cos(k * double(i)), -m});
    }
    return vr0s;
}

/**
 * @brief Multi-threading Bairstow's method (specific for auto-correlation
 * function)
 *
 * The function `pbairstow_autocorr` is implementing the Bairstow's method for
 * finding the roots of a real polynomial (specific for auto-correlation
 * function)
 *
 * @param[in] coeffs polynomial
 * @param[in,out] vrs vector of iterates
 * @param[in] options maximum iterations and tolorance
 * @return std::pair<unsigned int, bool>
 */
auto pbairstow_autocorr(const std::vector<double> &coeffs, std::vector<Vec2> &vrs,
                        const Options &options = Options()) -> std::pair<unsigned int, bool> {
    ThreadPool pool(std::thread::hardware_concurrency());

    const auto M = vrs.size();
    const auto rr = fun::Robin<size_t>(M);

    for (auto niter = 0U; niter != options.max_iters; ++niter) {
        auto tolerance = 0.0;
        std::vector<std::future<double>> results;
        for (auto i = 0U; i != M; ++i) {
            results.emplace_back(pool.enqueue([&, i]() {
                auto coeffs1 = coeffs;
                const auto N = coeffs.size() - 1;  // degree, assume even
                const auto &vri = vrs[i];
                auto vA = horner(coeffs1, N, vri);
                const auto tol_i = std::max(std::abs(vA.x()), std::abs(vA.y()));
                auto vA1 = horner(coeffs1, N - 2, vri);
                for (auto j : rr.exclude(i)) {  // exclude i
                    const auto vrj = vrs[j];    // make a copy, don't reference!
                    suppress(vA, vA1, vri, vrj);
                    const auto vrjn = numeric::Vector2<double>(-vrj.x(), 1.0) / vrj.y();
                    suppress(vA, vA1, vri, vrjn);
                }
                const auto vrin = numeric::Vector2<double>(-vri.x(), 1.0) / vri.y();
                suppress(vA, vA1, vri, vrin);

                vrs[i] -= delta(vA, vri, std::move(vA1));  // Gauss-Seidel fashion
                return tol_i;
            }));
        }
        for (auto &&result : results) {
            auto &&res = result.get();
            if (tolerance < res) {
                tolerance = res;
            }
        }
        if (tolerance < options.tolerance) {
            return {niter, true};
        }
    }
    return {options.max_iters, false};
}

/**
 * The function extracts the autocorrelation values from a given vector.
 *
 *   x^2 - r*x - q  or (-1/q) + (r/q) * x + x^2
 *   (x - a1)(x - a2) = x^2 - (a1 + a2) x + a1 * a2
 *
 *   x^2 + r*x + t or x^2 + (r/t) * x + (1/t)
 *   (x + a1)(x + a2) = x^2 + (a1 + a2) x + a1 * a2
 *
 * @param[in,out] vr The parameter `vr` is of type `Vec2`, which is a custom
 * class representing a 2D vector. It contains two components, `x` and `y`,
 * which are accessed using the `x()` and `y()` member functions respectively.
 */
void extract_autocorr(Vec2 &vr) {
    const auto &r = vr.x();
    const auto &q = vr.y();
    const auto hr = r / 2.0;
    const auto d = hr * hr + q;
    if (d < 0.0) {  // complex conjugate root
        if (q < -1.0) {
            vr = Vec2{-r, 1.0} / q;
        }
        // else no need to change
    } else {  // two real roots
        auto a1 = hr + (hr >= 0.0 ? sqrt(d) : -sqrt(d));
        auto a2 = -q / a1;
        if (std::abs(a1) > 1.0) {
            if (std::abs(a2) > 1.0) {
                a2 = 1.0 / a2;
            }
            a1 = 1.0 / a1;
            vr = Vec2{a1 + a2, -a1 * a2};
        } else if (std::abs(a2) > 1.0) {
            a2 = 1.0 / a2;
            vr = Vec2{a1 + a2, -a1 * a2};
        }
        // else no need to change
    }
}
