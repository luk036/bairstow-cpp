#pragma once

// import numpy as np
#include <utility>
#include <vector>

#include "matrix2.hpp"
#include "vector2.hpp"

using Vec2 = numeric::Vector2<double>;
using Mat2 = numeric::Matrix2<Vec2>;

class Options;

/**
 * @brief
 *
 * @param pa
 * @return std::vector<Vec2>
 */
extern auto initial_autocorr(const std::vector<double> &pa) -> std::vector<Vec2>;

/**
 * @brief
 *
 * @param pa
 * @param vrs
 * @param options
 * @return std::pair<unsigned int, bool>
 */
extern auto pbairstow_autocorr(const std::vector<double> &pa, std::vector<Vec2> &vrs,
                               const Options &options) -> std::pair<unsigned int, bool>;

/**
 * @brief
 *
 * @param vr
 */
extern void extract_autocorr(Vec2 &vr);
