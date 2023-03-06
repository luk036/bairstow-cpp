#pragma once

// import numpy as np
#include <complex>
#include <utility>
#include <vector>

class Options;

/**
 * @brief
 *
 * @param pa
 * @return std::vector<std::complex<double>>
 */
extern auto initial_aberth(const std::vector<double> &pa)
    -> std::vector<std::complex<double>>;

/**
 * @brief
 *
 * @param pa
 * @param zs
 * @param options
 * @return std::pair<unsigned int, bool>
 */
extern auto aberth(const std::vector<double> &pa,
                   std::vector<std::complex<double>> &zs,
                   const Options &options) -> std::pair<unsigned int, bool>;
