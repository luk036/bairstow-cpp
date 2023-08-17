#pragma once

// import numpy as np
// #include <gsl/gsl>
#include <utility>
#include <vector>

#include "matrix2.hpp"
#include "vector2.hpp"

using Vec2 = numeric::Vector2<double>;
using Mat2 = numeric::Matrix2<Vec2>;

/**
 * @brief
 *
 */
class Options {
  public:
    unsigned int max_iters = 2000U;
    double tol = 1e-14;
};

/**
 * @brief
 *
 * @param coeffs
 * @return std::vector<Vec2>
 */
extern auto initial_guess(const std::vector<double> &coeffs) -> std::vector<Vec2>;

/**
 * @brief
 *
 * @param coeffs
 * @param vrs
 * @param options
 * @return std::pair<unsigned int, bool>
 */
extern auto pbairstow_even(const std::vector<double> &coeffs,
                           std::vector<Vec2> &vrs, const Options &options)
    -> std::pair<unsigned int, bool>;

/**
 * @brief Horner's rule
 *
 * Horner's rule is a method for evaluating a polynomial of degree degree at a given
 * point x. It involves rewriting the polynomial as a nested multiplication and
 * addition of the form:
 *
 *  P(x) = a_0 + x(a_1 + x(a_2 + ... + x(a_{degree-1} + x(a_n))...))
 *
 * This form allows for efficient evaluation of the polynomial at a given point
 * x using only degree multiplications and degree additions. Horner's rule is commonly
 * used in numerical methods for polynomial evaluation and interpolation.
 *
 * @param coeffs1
 * @param degree
 * @param vr
 * @return Vec2
 */
extern auto horner(std::vector<double> &coeffs1, std::size_t degree, const Vec2 &vr)
    -> Vec2;

/**
 * @brief zero suppression
 *
 * zero suppression is a technique used in the Bairstow method to find the
 * coefficients of the linear remainder of a deflated polynomial without
 * explicitly constructing the deflated polynomial. The goal of zero suppression
 * is to perform the Bairstow process without the need for complex arithmetic
 * within iterations. The technique involves finding the coefficients of the
 * linear remainder of the deflated polynomial using the coefficients of the
 * linear remainder of the original polynomial and the known factor of the
 * original polynomial.
 *
 * @param[in,out] vA
 * @param[in,out] vA1
 * @param[in] vri
 * @param[in] vrj
 */
extern auto suppress(Vec2 &vA, Vec2 &vA1, const Vec2 &vri, const Vec2 &vrj) -> void;

/**
 * @brief
 *
 * @param[in] vr
 * @param[in] vp
 * @return Mat2
 */
inline auto makeadjoint(const Vec2 &vr, Vec2 &&vp) -> Mat2 {
    // auto &&[r, t] = vr;
    // auto &&[p, m] = vp;
    auto &&p = vp.x();
    auto &&s = vp.y();
    return {Vec2{s, -p}, Vec2{-p * vr.y(), p * vr.x() + s}};
}

/**
 * @brief
 *
 * @param[in] vA
 * @param[in] vr
 * @param[in] vp
 * @return Mat2
 */
inline auto delta(const Vec2 &vA, const Vec2 &vr, Vec2 &&vp) -> Vec2 {
    const auto mp = makeadjoint(vr, std::move(vp));  // 2 mul's
    return mp.mdot(vA) / mp.det();                   // 6 mul's + 2 div's
}

/**
 * @brief
 *
 * @param[in,out] coeffs1
 * @param[in] degree
 * @param[in] r
 * @return double
 */
inline auto horner_eval(std::vector<double> coeffs1, std::size_t degree, const double &z)
    -> double {
    for (auto i = 0U; i != degree; ++i) {
        coeffs1[i + 1] += coeffs1[i] * z;
    }
    return coeffs1[degree];
}
