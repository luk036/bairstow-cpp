#pragma once

// import numpy as np
#include <complex>
#include <utility>
#include <vector>

class Options;

/**
 * @brief Initial guess for the Aberth-Ehrlich method
 *
 * The `initial_aberth` function calculates the initial values for the Aberth-Ehrlich method for
 * finding the roots of a polynomial.
 *
 * @param[in] coeffs The `coeffs` parameter is a vector of doubles that represents the coefficients
 * of a polynomial.
 *
 * @return The function `initial_aberth` returns a vector of Complex numbers.
 */
extern auto initial_aberth(const std::vector<double> &coeffs) -> std::vector<std::complex<double>>;

/**
 * @brief Multi-threading Aberth-Ehrlich method
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
 * @param[in] zs `zs` is a vector of complex numbers representing the initial guesses for the roots
 * of the polynomial. The function will update these values iteratively to converge to the actual
 * roots.
 * @param[in] options The `options` parameter is an object of type `Options` that contains the
 * maximum number of iterations (`max_iters`) and the tolerance (`tol`). These options control the
 * convergence criteria for the Aberth-Ehrlich method.
 *
 * @return The `aberth` function returns a `std::pair<unsigned int, bool>`. The first element of the
 * pair represents the number of iterations performed, and the second element represents whether the
 * method converged to a solution within the specified tolerance.
 */
extern auto aberth(const std::vector<double> &coeffs, std::vector<std::complex<double>> &zs,
                   const Options &options) -> std::pair<unsigned int, bool>;
