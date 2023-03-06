#include <bairstow/ThreadPool.h> // for ThreadPool

#include <bairstow/rootfinding.hpp> // for Options
#include <cmath>                    // for acos, cos, sin
#include <complex>                  // for complex, operator*, operator+
#include <functional>               // for __base
#include <future>                   // for future
#include <py2cpp/range.hpp>         // for range
#include <thread>                   // for thread
#include <utility>                  // for pair
#include <vector>                   // for vector, vector<>::reference, __v...

using std::cos;
using std::sin;
using std::vector;
using Complex = std::complex<double>;

/**
 * @brief
 *
 * @param[in,out] coeffs
 * @param[in] n
 * @param[in] r
 * @return double
 */
template <typename C, typename Tp>
inline auto horner_eval_g(const C &coeffs, const Tp &z) -> Tp {
  Tp res = coeffs[0];
  for (auto i : py::range(1, coeffs.size())) {
    res = res * z + coeffs[i];
  }
  return res;
}

/**
 * @brief
 *
 * @param pa
 * @return vector<Complex>
 */
auto initial_aberth(const vector<double> &pa) -> vector<Complex> {
  static const auto TWO_PI = 2.0 * std::acos(-1.0);

  const auto n = pa.size() - 1;
  const auto c = -pa[1] / (double(n) * pa[0]);
  const auto Pc = horner_eval_g(pa, c);
  const auto re = std::pow(Complex(-Pc), 1.0 / double(n));
  const auto k = TWO_PI / double(n);
  auto z0s = vector<Complex>{};
  for (auto i : py::range(n)) {
    auto theta = k * (0.25 + double(i));
    auto z0 = c + re * Complex{std::cos(theta), std::sin(theta)};
    z0s.emplace_back(z0);
  }
  return z0s;
}

/**
 * @brief Multi-threading Bairstow's method (even degree only)
 *
 * @param[in] pa polynomial
 * @param[in,out] zs vector of iterates
 * @param[in] options maximum iterations and tolorance
 * @return std::pair<unsigned int, bool>
 */
auto aberth(const vector<double> &pa, vector<Complex> &zs,
            const Options &options = Options())
    -> std::pair<unsigned int, bool> {
  const auto m = zs.size();
  const auto n = pa.size() - 1; // degree, assume even
  auto converged = vector<bool>(m, false);
  auto coeffs = vector<double>(n);
  for (auto i : py::range(n)) {
    coeffs[i] = double(n - i) * pa[i];
  }
  ThreadPool pool(std::thread::hardware_concurrency());

  for (auto niter : py::range(options.max_iter)) {
    auto tol = 0.0;
    vector<std::future<double>> results;

    for (auto i : py::range(m)) {
      if (converged[i]) {
        continue;
      }
      results.emplace_back(pool.enqueue([&, i]() {
        const auto &zi = zs[i];
        const auto P = horner_eval_g(pa, zi);
        const auto tol_i = std::abs(P);
        if (tol_i < 1e-15) { // tunable
          converged[i] = true;
          return tol_i;
        }
        auto P1 = horner_eval_g(coeffs, zi);
        size_t j = 0;
        for (const auto &zj : zs) {
          if (j != i) {
            P1 -= P / (zi - zj);
          }
          ++j;
        }
        zs[i] -= P / P1; // Gauss-Seidel fashion
        return tol_i;
      }));
    }
    for (auto &&result : results) {
      auto &&res = result.get();
      if (tol < res) {
        tol = res;
      }
    }
    if (tol < options.tol) {
      return {niter, true};
    }
  }
  return {options.max_iter, false};
}
