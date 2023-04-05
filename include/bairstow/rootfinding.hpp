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
  unsigned int max_iter = 2000U;
  double tol = 1e-14;
};

/**
 * @brief
 *
 * @param pa
 * @return std::vector<Vec2>
 */
extern auto initial_guess(const std::vector<double> &pa) -> std::vector<Vec2>;

/**
 * @brief
 *
 * @param pa
 * @param vrs
 * @param options
 * @return std::pair<unsigned int, bool>
 */
extern auto pbairstow_even(const std::vector<double> &pa,
                           std::vector<Vec2> &vrs, const Options &options)
    -> std::pair<unsigned int, bool>;

/**
 * @brief
 *
 * @param pb
 * @param n
 * @param vr
 * @return Vec2
 */
extern auto horner(std::vector<double> &pb, std::size_t n, const Vec2 &vr)
    -> Vec2;

/**
 * @brief zero suppression
 *
 * @param[in,out] vA
 * @param[in,out] vA1
 * @param[in] vri
 * @param[in] vrj
 */
extern auto suppress(Vec2 &vA, Vec2 &vA1, const Vec2 &vri, const Vec2 &vrj)
    -> void;

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
  const auto mp = makeadjoint(vr, std::move(vp)); // 2 mul's
  return mp.mdot(vA) / mp.det();                  // 6 mul's + 2 div's
}

/**
 * @brief
 *
 * @param[in,out] pb
 * @param[in] n
 * @param[in] r
 * @return double
 */
inline auto horner_eval(std::vector<double> pb, std::size_t n, const double &z)
    -> double {
  for (auto i = 0U; i != n; ++i) {
    pb[i + 1] += pb[i] * z;
  }
  return pb[n];
}
