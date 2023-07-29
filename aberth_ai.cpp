#include "Options.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <rayon/iter/par_iter.h>
#include <vector>

const double TWO_PI = 2 * M_PI;

std::complex<double> horner_eval_c(const std::vector<double> &coeffs,
                                   const std::complex<double> &zval) {
    std::complex<double> result = 0.0;
    for (const auto &coeff : coeffs) {
        result = result * zval + std::complex<double>(coeff, 0.0);
    }
    return result;
}

std::pair<size_t, bool> aberth_mt(const std::vector<double> &coeffs,
                                  std::vector<std::complex<double>> &zs,
                                  const Options &options) {
    size_t m_rs = zs.size();
    size_t degree = coeffs.size() - 1;
    std::vector<double> pb(degree);
    for (size_t i = 0; i < degree; i++) {
        pb[i] = coeffs[i] * (degree - i);
    }
    std::vector<std::complex<double>> zsc(m_rs);
    std::vector<bool> converged(m_rs);
    for (size_t niter = 0; niter < options.max_iters; niter++) {
        double tol = 0.0;
        zsc = zs;
        double tol_i = std::transform_reduce(
            zs.begin(), zs.end(), converged.begin(), 0.0,
            [](double x, double y) { return std::max(x, y); },
            [&](const std::complex<double> &zi, bool &converged) {
                return aberth_job(coeffs, zi, converged, zsc, pb);
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
