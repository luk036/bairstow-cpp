#pragma once

// import numpy as np
#include <complex>
#include <utility>
#include <vector>

class Options;

/**
 * @brief
 *
 * @param coeffs
 * @return std::vector<std::complex<double>>
 */
extern auto initial_aberth(const std::vector<double> &coeffs)
    -> std::vector<std::complex<double>>;

/**
 * @brief
 *
 * Aberth's method is a method for finding the roots of a polynomial that is
 * robust but requires complex arithmetic even if the polynomial is real. This
 * is because it starts with complex initial approximations.
 *
 * @param coeffs
 * @param zs
 * @param options
 * @return std::pair<unsigned int, bool>
 */
extern auto aberth(const std::vector<double> &coeffs,
                   std::vector<std::complex<double>> &zs,
                   const Options &options) -> std::pair<unsigned int, bool>;
