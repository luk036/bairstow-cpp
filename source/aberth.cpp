#include <bairstow/ThreadPool.h>  // for ThreadPool

#include <bairstow/robin.hpp>        // for Robin
#include <bairstow/rootfinding.hpp>  // for Options
#include <cmath>                     // for acos, cos, sin
#include <complex>                   // for complex, operator*, operator+
#include <future>                    // for future
#include <numeric>                   // for accumulate, partial_sum
#include <utility>                   // for pair
#include <vector>                    // for vector, vector<>::reference, __v...

using std::cos;
using std::sin;
using std::vector;
using Complex = std::complex<double>;

/**
 * The function `horner_eval_g` is implementing the Horner's method for
 * evaluating a polynomial at a given point.
 *
 * @param[in] coeffs The `coeffs` parameter is a vector representing the coefficients of a
 * polynomial. Each element of the vector corresponds to a term in the polynomial, starting from the
 * highest degree term and ending with the constant term.
 * @param[in] z
 * @return Tp
 */
template <typename C, typename Tp> inline auto horner_eval_g(const C &coeffs, const Tp &z) -> Tp {
    // return std::accumulate(coeffs.cbegin(), coeffs.cend(),
    //                        [&z](Tp &res, auto &c) { return res * z + Tp(c); });
    auto itr = coeffs.cbegin();
    auto res = Tp(*itr++);
    for (; itr != coeffs.cend(); ++itr) {
        res = res * z + *itr;
    }
    // for (auto i = 1U; i != coeffs.size(); ++i) {
    //     res = res * z + coeffs[i];
    // }
    return res;
}

/**
 * @brief Initial guess for the Aberth-Ehrlich method
 *
 * The `initial_aberth` function calculates the initial values for the Aberth-Ehrlich method for
 * finding the roots of a polynomial.
 *
 * @param[in] coeffs The `coeffs` parameter is a vector representing the coefficients of a
 * polynomial. Each element of the vector corresponds to a term in the polynomial, starting from the
 * highest degree term and ending with the constant term.
 *
 * @return The function `initial_aberth` returns a vector of Complex numbers representing the
 * initial guesses for the roots of the polynomial.
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
 * @brief Single-threading Aberth-Ehrlich method
 *
 * The `aberth` function is a multi-threaded implementation of the Aberth-Ehrlich method for finding
 * the roots of a polynomial.
 *
 * Aberth's method is a method for finding the roots of a polynomial that is
 * robust but requires complex arithmetic even if the polynomial is real. This
 * is because it starts with complex initial approximations.
 *
 * @param[in] coeffs The `coeffs` parameter is a vector representing the coefficients of a
 * polynomial. Each element of the vector corresponds to a term in the polynomial, starting from the
 * highest degree term and ending with the constant term. For example, if the polynomial is `3x^2 +
 * 2x +
 * @param[in,out] zs `zs` is a vector of complex numbers representing the initial guesses for the
 * roots of the polynomial. The function will update these values iteratively to converge to the
 * actual roots.
 * @param[in] options The `options` parameter is an object of type `Options` that contains the
 * maximum number of iterations (`max_iters`) and the tolerance (`tolerance`). These options control
 * the convergence criteria for the Aberth-Ehrlich method.
 *
 * @return The `aberth` function returns a `std::pair<unsigned int, bool>`. The first element of the
 * pair represents the number of iterations performed, and the second element represents whether the
 * method converged to a solution within the specified tolerance.
 */
auto aberth(const vector<double> &coeffs, vector<Complex> &zs,
            const Options &options = Options()) -> std::pair<unsigned int, bool> {
    const auto m = zs.size();
    const auto degree = coeffs.size() - 1;  // degree, assume even
    const auto rr = fun::Robin<size_t>(m);
    auto coeffs1 = vector<double>(degree);
    for (auto i = 0U; i != degree; ++i) {
        coeffs1[i] = double(degree - i) * coeffs[i];
    }

    for (auto niter = 0U; niter != options.max_iters; ++niter) {
        auto tolerance = 0.0;
        vector<std::future<double>> results;

        for (auto i = 0U; i != m; ++i) {
            auto do_core = [&, i]() -> double {
                const auto &zi = zs[i];
                const auto P = horner_eval_g(coeffs, zi);
                const auto tol_i = std::abs(P);
                auto P1 = horner_eval_g(coeffs1, zi);
                for (auto j : rr.exclude(i)) {
                    P1 -= P / (zi - zs[j]);
                }
                zs[i] -= P / P1;  // Gauss-Seidel fashion
                return tol_i;
            };
            auto res = do_core();
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
 * @brief Multi-threading Aberth-Ehrlich method
 *
 * The `aberth_mt` function is a multi-threaded implementation of the Aberth-Ehrlich method for
 * finding the roots of a polynomial.
 *
 * Aberth's method is a method for finding the roots of a polynomial that is
 * robust but requires complex arithmetic even if the polynomial is real. This
 * is because it starts with complex initial approximations.
 *
 * @param[in] coeffs The `coeffs` parameter is a vector representing the coefficients of a
 * polynomial. Each element of the vector corresponds to a term in the polynomial, starting from the
 * highest degree term and ending with the constant term. For example, if the polynomial is `3x^2 +
 * 2x +
 * @param[in,out] zs `zs` is a vector of complex numbers representing the initial guesses for the
 * roots of the polynomial. The function will update these values iteratively to converge to the
 * actual roots.
 * @param[in] options The `options` parameter is an object of type `Options` that contains the
 * maximum number of iterations (`max_iters`) and the tolerance (`tolerance`). These options control
 * the convergence criteria for the Aberth-Ehrlich method.
 *
 * @return The `aberth` function returns a `std::pair<unsigned int, bool>`. The first element of the
 * pair represents the number of iterations performed, and the second element represents whether the
 * method converged to a solution within the specified tolerance.
 */
auto aberth_mt(const vector<double> &coeffs, vector<Complex> &zs,
               const Options &options = Options()) -> std::pair<unsigned int, bool> {
    ThreadPool pool(std::thread::hardware_concurrency());

    const auto m = zs.size();
    const auto degree = coeffs.size() - 1;  // degree, assume even
    const auto rr = fun::Robin<size_t>(m);
    auto coeffs1 = vector<double>(degree);
    for (auto i = 0U; i != degree; ++i) {
        coeffs1[i] = double(degree - i) * coeffs[i];
    }

    for (auto niter = 0U; niter != options.max_iters; ++niter) {
        auto tolerance = 0.0;
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
                zs[i] -= P / P1;  // Gauss-Seidel fashion
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
