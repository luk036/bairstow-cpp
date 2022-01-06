#include <bairstow/ThreadPool.h>

#include <bairstow/rootfinding.hpp>
#include <cmath>    // import pow, cos, sqrt
#include <complex>  // import pow, exp

/**
 * @brief
 *
 * @param[in,out] pb
 * @param[in] n
 * @param[in] r
 * @return double
 */
template <typename C, typename Tp> inline auto horner_eval_g(const C& pb, const Tp& z) -> Tp {
    Tp ans = pb[0];
    for (auto i = 1U; i != pb.size(); ++i) {
        ans = ans * z + pb[i];
    }
    return ans;
}

/**
 * @brief
 *
 * @param[in] pa
 * @return std::vector<vec2>
 */
auto initial_aberth(const std::vector<double>& pa) -> std::vector<std::complex<double>> {
    static const auto PI = std::acos(-1.);

    const auto N = int(pa.size()) - 1;
    const auto c = -pa[1] / (N * pa[0]);
    const auto Pc = horner_eval_g(pa, c);
    const auto re = std::pow(std::complex<double>(-Pc), 1. / N);
    const auto k = 2 * PI / N;
    auto z0s = std::vector<std::complex<double>>{};
    for (auto i = 0; i < N; ++i) {
        auto theta = k * (i + 0.25);
        auto z0 = c + re * std::complex<double>{std::cos(theta), std::sin(theta)};
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
 * @return std::tuple<unsigned int, bool>
 */
auto aberth(const std::vector<double>& pa, std::vector<std::complex<double>>& zs,
            const Options& options = Options()) -> std::tuple<unsigned int, bool> {
    const auto M = zs.size();
    const auto N = int(pa.size()) - 1;  // degree, assume even
    auto found = false;
    auto converged = std::vector<bool>(M, false);
    auto pb = std::vector<double>(N);
    for (auto i = 0; i < N; ++i) {
        pb[i] = (N - i) * pa[i];
    }
    auto niter = 1U;
    ThreadPool pool(std::thread::hardware_concurrency());

    for (; niter != options.max_iter; ++niter) {
        auto tol = 0.0;
        std::vector<std::future<double>> results;

        for (auto i = 0U; i != M; ++i) {
            if (converged[i]) {
                continue;
            }
            results.emplace_back(pool.enqueue([&, i]() {
                const auto& zi = zs[i];
                const auto P = horner_eval_g(pa, zi);
                const auto tol_i = std::abs(P);
                if (tol_i < 1e-15) {
                    converged[i] = true;
                    return tol_i;
                }
                auto P1 = horner_eval_g(pb, zi);
                for (auto j = 0U; j != M; ++j) {  // exclude i
                    if (j == i) {
                        continue;
                    }
                    const auto zj = zs[j];  // make a copy, don't reference!
                    P1 -= P / (zi - zj);
                }
                zs[i] -= P / P1;  // Gauss-Seidel fashion
                return tol_i;
            }));
        }
        for (auto&& result : results) {
            tol = std::max(tol, result.get());
        }
        if (tol < options.tol) {
            found = true;
            break;
        }
    }
    return {niter, found};
}
