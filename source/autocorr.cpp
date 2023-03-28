#include <bairstow/ThreadPool.h> // for ThreadPool

// #include <__bit_reference>           // for __bit_reference
#include <bairstow/autocorr.hpp>    // for extract_autocorr, initial_autocorr
#include <bairstow/robin.hpp>       // for Robin
#include <bairstow/rootfinding.hpp> // for Vec2, delta, horner, Options
#include <bairstow/vector2.hpp>     // for Vector2, operator-, operator/
#include <cmath>                    // for abs, sqrt, acos, cos, pow
#include <functional>               // for __base
#include <future>                   // for future
#include <thread>                   // for thread
#include <type_traits>              // for move
#include <utility>                  // for pair
#include <vector>                   // for vector, vector<>::reference, __v...

/**
 * @brief initial guess (specific for auto-correlation function)
 *
 * @param[in] pa
 * @return std::vector<Vec2>
 */
auto initial_autocorr(const std::vector<double> &pa) -> std::vector<Vec2> {
  static const auto PI = std::acos(-1.);

  auto degree = pa.size() - 1;
  const auto re = std::pow(std::abs(pa[degree]), 1.0 / double(degree));

  degree /= 2;
  const auto k = PI / double(degree);
  const auto m = re * re;
  auto vr0s = std::vector<Vec2>{};
  for (auto i = 1U; i < degree; i += 2) {
    vr0s.emplace_back(Vec2{2 * re * std::cos(k * double(i)), -m});
  }
  return vr0s;
}

/**
 * @brief Multi-threading Bairstow's method (specific for auto-correlation
 * function)
 *
 * @param[in] pa polynomial
 * @param[in,out] vrs vector of iterates
 * @param[in] options maximum iterations and tolorance
 * @return std::pair<unsigned int, bool>
 */
auto pbairstow_autocorr(const std::vector<double> &pa, std::vector<Vec2> &vrs,
                        const Options &options = Options())
    -> std::pair<unsigned int, bool> {
  ThreadPool pool(std::thread::hardware_concurrency());

  const auto M = vrs.size();
  const auto rr = fun::Robin<size_t>(M);

  for (auto niter = 0U; niter != options.max_iter; ++niter) {
    auto tol = 0.0;
    std::vector<std::future<double>> results;
    for (auto i = 0U; i != M; ++i) {
      results.emplace_back(pool.enqueue([&, i]() -> double {
        auto pb = pa;
        const auto N = pa.size() - 1; // degree, assume even
        const auto &vri = vrs[i];
        auto vA = horner(pb, N, vri);
        const auto tol_i = std::max(std::abs(vA.x()), std::abs(vA.y()));
        auto vA1 = horner(pb, N - 2, vri);
        for (auto j : rr.exclude(i)) { // exclude i
          const auto vrj = vrs[j];     // make a copy, don't reference!
          suppress(vA, vA1, vri, vrj);
          const auto vrjn = numeric::Vector2<double>(-vrj.x(), 1.0) / vrj.y();
          suppress(vA, vA1, vri, vrjn);
        }
        const auto vrin = numeric::Vector2<double>(-vri.x(), 1.0) / vri.y();
        suppress(vA, vA1, vri, vrin);

        vrs[i] -= delta(vA, vri, std::move(vA1)); // Gauss-Seidel fashion
        return tol_i;
      }));
    }
    for (auto &&result : results) {
      auto &&res = result.get();
      if (tol < res) {
        tol = res;
      }
    }
    // fmt::print("tol: {}\n", tol);
    if (tol < options.tol) {
      return {niter, true};
    }
  }
  return {options.max_iter, false};
}

/**
 * @brief Extract the quadratic function where its roots are within a unit
 * circle
 *
 *   x^2 - r*x - q  or (-1/q) + (r/q) * x + x^2
 *   (x - a1)(x - a2) = x^2 - (a1 + a2) x + a1 * a2
 *
 *   x^2 + r*x + t or x^2 + (r/t) * x + (1/t)
 *   (x + a1)(x + a2) = x^2 + (a1 + a2) x + a1 * a2
 *
 * @param[in,out] vr
 */
void extract_autocorr(Vec2 &vr) {
  const auto &r = vr.x();
  const auto &q = vr.y();
  const auto hr = r / 2.0;
  const auto d = hr * hr + q;
  if (d < 0.0) { // complex conjugate root
    if (q < -1.0) {
      vr = Vec2{-r, 1.0} / q;
    }
    // else no need to change
  } else { // two real roots
    auto a1 = hr + (hr >= 0.0 ? sqrt(d) : -sqrt(d));
    auto a2 = -q / a1;
    if (std::abs(a1) > 1.0) {
      if (std::abs(a2) > 1.0) {
        a2 = 1.0 / a2;
      }
      a1 = 1.0 / a1;
      vr = Vec2{a1 + a2, -a1 * a2};
    } else if (std::abs(a2) > 1.0) {
      a2 = 1.0 / a2;
      vr = Vec2{a1 + a2, -a1 * a2};
    }
    // else no need to change
  }
}
