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
 * The function `horner` implements the Horner's method for evaluating a
 * polynomial at a given point.
 *
 * @param[in, out] coeffs1 coeffs1 is a reference to a vector of doubles. It is used to
 * store the coefficients of a polynomial.
 * @param[in] degree The parameter `degree` represents the size of the vector `coeffs1`. It
 * indicates the number of elements in the vector `coeffs1`.
 * @param[in] vr vr is a Vec2 object, which represents a 2D vector. It has two
 * components, vr.x() and vr.y(), which are used in the calculations inside the
 * horner function.
 *
 * @return a Vec2 object.
 */
auto horner(std::vector<double> &coeffs1, size_t degree, const Vec2 &vr) -> Vec2 {
    for (auto i = 0U; i != degree - 1; ++i) {
        coeffs1[i + 1] += coeffs1[i] * vr.x();
        coeffs1[i + 2] += coeffs1[i] * vr.y();
    }
    return Vec2{coeffs1[degree - 1], coeffs1[degree]};
}

/**
 * The function `suppress` calculates and updates the values of `vA` and `vA1`
 * based on the given input vectors `vri` and `vrj`.
 *
 * @param[in, out] vA A reference to a Vec2 object representing vector A.
 * @param[in, out] vA1 vA1 is a reference to a Vec2 object.
 * @param[in] vri A vector representing the position of point i.
 * @param[in] vrj The parameter `vrj` represents a `Vec2` object.
 */
auto suppress(Vec2 &vA, Vec2 &vA1, const Vec2 &vri, const Vec2 &vrj) -> void {
    // const auto [r, q] = vri;
    // const auto [p, s] = vri - vrj;
    const auto vp = vri - vrj;
    auto &&p = vp.x();
    auto &&s = vp.y();
    const auto M = Mat2{Vec2{s, -p}, Vec2{-p * vri.y(), p * vri.x() + s}};
    const auto e = M.det();
    vA = M.mdot(vA) / e;

    const auto vd = vA1 - vA;
    const auto vc = Vec2{vd.x(), vd.y() - vA.x() * p};
    vA1 = M.mdot(vc) / e;
}

/**
 * @brief Initial guess for the parallel Bairstow method
 *
 * The `initial_guess` function calculates the initial values for the parallel Bairstow method for
 * finding the roots of a real polynomial.
 *
 * @param[in] coeffs coeffs is a vector of doubles that represents the coefficients of a polynomial.
 *
 * @return The function `initial_guess` returns a vector of `Vec2` objects.
 */
auto initial_guess(const std::vector<double> &coeffs) -> std::vector<Vec2> {
    auto N = coeffs.size() - 1;
    const auto c = -coeffs[1] / (double(N) * coeffs[0]);
    auto coeffs1 = coeffs;
    const auto Pc = horner_eval(coeffs1, N, c);  // TODO
    const auto re = std::pow(std::abs(Pc), 1.0 / double(N));
    N /= 2;
    N *= 2;  // make even
    const auto k = M_PI / double(N);
    const auto m = c * c + re * re;
    auto vr0s = std::vector<Vec2>{};
    for (auto i = 1U; i < N; i += 2) {
        const auto temp = re * std::cos(k * i);
        auto r0 = 2 * (c + temp);
        auto t0 = -(m + 2 * c * temp);
        vr0s.emplace_back(Vec2{std::move(r0), std::move(t0)});
    }
    return vr0s;
}

/**
 * @brief Multi-threading Bairstow's method (even degree only)
 *
 * The function `pbairstow_even` is implementing the Bairstow's method for
 * finding the roots of a real polynomial with an even degree.
 *
 * @param[in] coeffs polynomial
 * @param[in,out] vrs vector of iterates
 * @param[in] options maximum iterations and tolorance
 * @return std::pair<unsigned int, bool>
 */
auto pbairstow_even(const std::vector<double> &coeffs, std::vector<Vec2> &vrs,
                    const Options &options = Options()) -> std::pair<unsigned int, bool> {
    ThreadPool pool(std::thread::hardware_concurrency());

    // const auto degree = coeffs.size() - 1; // degree, assume even
    const auto M = vrs.size();
    const auto rr = fun::Robin<size_t>(M);

    for (auto niter = 0U; niter != options.max_iters; ++niter) {
        auto tol = 0.0;
        std::vector<std::future<double>> results;

        for (auto i = 0U; i != M; ++i) {
            results.emplace_back(pool.enqueue([&, i]() -> double {
                const auto degree = coeffs.size() - 1;  // degree, assume even
                const auto &vri = vrs[i];
                auto coeffs1 = coeffs;
                auto vA = horner(coeffs1, degree, vri);
                auto vA1 = horner(coeffs1, degree - 2, vri);
                const auto tol_i = std::max(std::abs(vA.x()), std::abs(vA.y()));
                for (auto j : rr.exclude(i)) {
                    const auto vrj = vrs[j];  // make a copy, don't reference!
                    suppress(vA, vA1, vri, vrj);
                }
                vrs[i] -= delta(vA, vri, std::move(vA1));  // Gauss-Seidel fashion
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

// auto find_rootq(const Vec2& r) {
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
