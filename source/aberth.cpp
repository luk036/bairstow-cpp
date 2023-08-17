#include <bairstow/ThreadPool.h>    // for ThreadPool
#include <bairstow/robin.hpp>       // for Robin
#include <bairstow/rootfinding.hpp> // for Options

#include <cmath>      // for acos, cos, sin
#include <complex>    // for complex, operator*, operator+
#include <functional> // for __base
#include <future>     // for future
#include <thread>     // for thread
#include <utility>    // for pair
#include <vector>     // for vector, vector<>::reference, __v...

using std::cos;
using std::sin;
using std::vector;
using Complex = std::complex<double>;

/**
 * The function `horner_eval_g` is implementing the Horner's method for
 * evaluating a polynomial at a given point.
 *
 * @param[in,out] coeffs
 * @param[in] degree
 * @param[in] r
 * @return double
 */
template <typename C, typename Tp>
inline auto horner_eval_g(const C &coeffs, const Tp &z) -> Tp {
    Tp res = coeffs[0];
    for (auto i = 1U; i != coeffs.size(); ++i) {
        res = res * z + coeffs[i];
    }
    return res;
}

/**
 * The function calculates the initial values for the Aberth-Ehrlich method for
 * finding the roots of a polynomial.
 *
 * @param[in] coeffs The parameter `coeffs` is a vector of doubles.
 *
 * @return The function `initial_aberth` returns a vector of Complex numbers.
 */
auto initial_aberth(const vector<double> &coeffs) -> vector<Complex> {
    static const auto TWO_PI = 2.0 * std::acos(-1.0);

    const auto degree = coeffs.size() - 1;
    const auto c = -coeffs[1] / (double(degree) * coeffs[0]);
    const auto Pc = horner_eval_g(coeffs, c);
    const auto re = std::pow(Complex(-Pc), 1.0 / double(degree));
    const auto k = TWO_PI / double(degree);
    auto z0s = vector<Complex>{};
    for (auto i = 0U; i != degree; ++i) {
        auto theta = k * (0.25 + double(i));
        auto z0 = c + re * Complex{std::cos(theta), std::sin(theta)};
        z0s.emplace_back(z0);
    }
    return z0s;
}

/**
 * @brief Multi-threading Aberth-Ehrlich method
 *
 * The `aberth` function is an implementation of the Aberth-Ehrlich method for
 * finding the roots of a polynomial.
 *
 * @param[in] coeffs polynomial
 * @param[in,out] zs vector of iterates
 * @param[in] options maximum iterations and tolorance
 * @return std::pair<unsigned int, bool>
 */
auto aberth(const vector<double> &coeffs, vector<Complex> &zs,
            const Options &options = Options())
    -> std::pair<unsigned int, bool> {
    ThreadPool pool(std::thread::hardware_concurrency());

    const auto m = zs.size();
    const auto degree = coeffs.size() - 1; // degree, assume even
    const auto rr = fun::Robin<size_t>(m);
    auto coeffs1 = vector<double>(degree);
    for (auto i = 0U; i != degree; ++i) {
        coeffs1[i] = double(degree - i) * coeffs[i];
    }

    for (auto niter = 0U; niter != options.max_iters; ++niter) {
        auto tol = 0.0;
        vector<std::future<double>> results;

        for (auto i = 0U; i != m; ++i) {
            results.emplace_back(pool.enqueue([&, i]() -> double {
                const auto &zi = zs[i];
                const auto P = horner_eval_g(coeffs, zi);
                const auto tol_i = std::abs(P);
                auto P1 = horner_eval_g(coeffs1, zi);
                for (auto j : rr.exclude(i)) {
                    P1 -= P / (zi - zs[j]);
                }
                zs[i] -= P / P1; // Gauss-Seidel fashion
                return tol_i;
            }));
        }
        for (auto &&result : results) {
            auto &&res = result.get();
            if (tol < res) {
                tol = res;
            }
        }
        if (tol < options.tol) {
            return {niter, true};
        }
    }
    return {options.max_iters, false};
}
