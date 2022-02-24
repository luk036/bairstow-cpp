#pragma once

// import numpy as np
#include <utility>
#include <vector>

#include "matrix2.hpp"
#include "vector2.hpp"

using vec2 = numeric::vector2<double>;
using mat2 = numeric::matrix2<vec2>;

class Options;

extern auto initial_autocorr(const std::vector<double>& pa) -> std::vector<vec2>;
extern auto pbairstow_autocorr(const std::vector<double>& pa, std::vector<vec2>& vrs,
                               const Options& options) -> std::pair<unsigned int, bool>;
extern void extract_autocorr(vec2& vr);
