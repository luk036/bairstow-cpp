#pragma once

// import numpy as np
#include <tuple>
#include <vector>

#include "matrix2.hpp"
#include "vector2.hpp"

using vec2 = numeric::vector2<double>;
using mat2 = numeric::matrix2<vec2>;

/**
 * @brief
 *
 * @param[in] vr
 * @param[in] vp
 * @return mat2
 */
inline auto makeadjoint(const vec2& vr, vec2&& vp) -> mat2 {
    const auto &r = vr.x(), q = vr.y();
    const auto &p = vp.x(), s = vp.y();
    return {vec2{s, -p}, vec2{-p * q, p * r + s}};
}

/**
 * @brief
 *
 * @param[in] vA
 * @param[in] vr
 * @param[in] vp
 * @return mat2
 */
inline auto delta(const vec2& vA, const vec2& vr, vec2&& vp) -> vec2 {
    const auto mp = makeadjoint(vr, std::move(vp));  // 2 mul's
    return mp.mdot(vA) / mp.det();                   // 6 mul's + 2 div's
}

// extern void suppress(const vec2& vA, vec2& vA1, const vec2& vr, const vec2& vrj);
// extern auto check_newton(const vec2& vA, const vec2& vA1, const vec2& vr) -> vec2;
// extern auto horner_eval(std::vector<double>& pb, std::size_t n, const double& r) -> double;
extern auto horner(std::vector<double>& pb, std::size_t n, const vec2& vr) -> vec2;

class Options {
  public:
    unsigned int max_iter = 2000U;
    double tol = 1e-14;
};

extern auto initial_guess(const std::vector<double>& pa) -> std::vector<vec2>;
extern auto pbairstow_even(const std::vector<double>& pa, std::vector<vec2>& vrs,
                           const Options& options) -> std::tuple<unsigned int, bool>;
