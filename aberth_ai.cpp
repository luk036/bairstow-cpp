#include <rayon/iter/par_iter.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <vector>

#include "Options.h"

const double TWO_PI = 2 * M_PI;

/**
 * The function `horner_eval_c` evaluates a polynomial with complex coefficients at a given complex
 * value using Horner's method.
 * 
 * @param coeffs The `coeffs` parameter is a vector of double values representing the coefficients of a
 * polynomial. Each element in the vector corresponds to a term in the polynomial, starting from the
 * highest degree term.
 * @param zval The parameter `zval` is a complex number that represents the value at which the
 * polynomial is evaluated.
 * 
 * @return The function `horner_eval_c` returns a complex number of type `std::complex<double>`.
 */
std::complex<double> horner_eval_c(const std::vector<double> &coeffs,
                                   const std::complex<double> &zval) {
    std::complex<double> result = 0.0;
    for (const auto &coeff : coeffs) {
        result = result * zval + std::complex<double>(coeff, 0.0);
    }
    return result;
}

/**
 * The function `aberth_mt` performs the Aberth method for finding the roots of a polynomial using
 * multiple threads.
 * 
 * @param coeffs The `coeffs` parameter is a vector of doubles representing the coefficients of a
 * polynomial. The polynomial is of degree `degree`, where `degree` is the size of `coeffs` minus 1.
 * @param zs A vector of complex numbers representing the initial guesses for the roots of a polynomial
 * equation.
 * @param options The `options` parameter is an object of type `Options`. It contains various options
 * for the Aberth method algorithm, such as the maximum number of iterations (`max_iters`) and the
 * tolerance (`tol`) for convergence.
 * 
 * @return The function `aberth_mt` returns a `std::pair<size_t, bool>`. The first element of the pair
 * represents the number of iterations performed by the algorithm, and the second element represents
 * whether the algorithm converged or not.
 */
std::pair<size_t, bool> aberth_mt(const std::vector<double> &coeffs,
                                  std::vector<std::complex<double>> &zs,
                                  const Options &options) {
    size_t m_zs = zs.size();
    size_t degree = coeffs.size() - 1;
    std::vector<double> coeffs1(degree);
    for (size_t i = 0; i < degree; i++) {
        coeffs1[i] = coeffs[i] *double (degree - i);
    }
    std::vector<std::complex<double>> zsc(m_zs);
    std::vector<bool> converged(m_zs);
    for (size_t niter = 0; niter < options.max_iters; niter++) {
        double tol = 0.0;
        zsc = zs;
        double tol_i = std::transform_reduce(
            zs.begin(), zs.end(), converged.begin(), 0.0,
            [](double x, double y) { return std::max(x, y); },
            [&](const std::complex<double> &zi, bool &converged) {
                return aberth_job(coeffs, zi, converged, zsc, coeffs1);
            });
        if (tol < tol_i) {
            tol = tol_i;
        }
        if (tol < options.tol) {
            return std::make_pair(niter, true);
        }
    }
    return std::make_pair(options.max_iters, false);
}
