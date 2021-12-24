#include <bairstow/ThreadPool.h>

#include <bairstow/rootfinding.hpp>
#include <cmath>  // import pow, cos, sqrt

// using vec2 = numeric::vector2<double>;
// using mat2 = numeric::matrix2<vec2>;

/**
 * @brief
 *
 * @param vA
 * @param vA1
 * @param vr
 * @param vrj
 */
void suppress(const vec2& vA, vec2& vA1, const vec2& vr, const vec2& vrj) {
    auto mp = makeadjoint(vrj, vr - vrj);  // 2 mul's
    vA1 -= mp.mdot(vA) / mp.det();         // 6 mul's + 2 div's
}

/**
 * @brief
 *
 * @param[in,out] pb
 * @param[in] n
 * @param[in] r
 * @return double
 */
auto horner_eval(std::vector<double>& pb, std::size_t n, const double& r) -> double {
    for (auto i = 0U; i != n; ++i) {
        pb[i + 1] += pb[i] * r;
    }
    return pb[n];
}

/**
 * @brief
 *
 * @param[in,out] pb
 * @param[in] n
 * @param[in] vr
 * @return vec2
 */
auto horner(std::vector<double>& pb, size_t n, const vec2& vr) -> vec2 {
    const auto &r = vr.x(), q = vr.y();
    pb[1] += pb[0] * r;
    for (auto i = 2U; i != n; ++i) {
        pb[i] += pb[i - 2] * q + pb[i - 1] * r;
    }
    pb[n] += pb[n - 2] * q;
    return vec2{pb[n - 1], pb[n]};
}

/**
 * @brief
 *
 * @param[in] pa
 * @return std::vector<vec2>
 */
auto initial_guess(const std::vector<double>& pa) -> std::vector<vec2> {
    static const auto PI = std::acos(-1.);

    auto N = pa.size() - 1;
    auto Nf = double(N);
    auto c = -pa[1] / (Nf * pa[0]);
    auto pb = pa;
    auto Pc = horner_eval(pb, N, c);  // ???
    auto re = std::pow(std::abs(Pc), 1. / Nf);
    auto k = PI / Nf;
    auto m = c * c + re * re;
    auto vr0s = std::vector<vec2>{};
    for (auto i = 1U; i < N; i += 2) {
        auto temp = re * std::cos(k * i);
        auto r0 = 2 * (c + temp);
        auto t0 = m + 2 * c * temp;
        vr0s.emplace_back(vec2{r0, -t0});
    }
    return vr0s;
}

/**
 * @brief Multi-threading Bairstow's method (even degree only)
 *
 * @param[in] pa polynomial
 * @param[in,out] vrs vector of iterates
 * @param[in] options maximum iterations and tolorance
 * @return std::tuple<unsigned int, bool>
 */
auto pbairstow_even(const std::vector<double>& pa, std::vector<vec2>& vrs,
                    const Options& options = Options()) -> std::tuple<unsigned int, bool> {
    auto N = pa.size() - 1;  // degree, assume even
    auto M = vrs.size();
    auto found = false;
    auto converged = std::vector<bool>(M, false);
    auto niter = 1U;
    ThreadPool pool(std::thread::hardware_concurrency());

    for (; niter != options.max_iter; ++niter) {
        auto tol = 0.0;
        std::vector<std::future<double>> results;

        for (auto i = 0U; i != M && !converged[i]; ++i) {
            results.emplace_back(pool.enqueue([&, i]() {
                auto pb = pa;
                // auto n = pa.size() - 1;
                const auto& vri = vrs[i];
                const auto vA = horner(pb, N, vri);
                auto tol_i = vA.norm_inf();
                if (tol_i < options.tol) {
                    converged[i] = true;
                    // continue;
                    return tol_i;
                }
                // tol = std::max(tol, toli);
                auto vA1 = horner(pb, N - 2, vri);
                for (auto j = 0U; j != M && j != i; ++j) {  // exclude i
                    auto vrj = vrs[j];                      // make a copy, don't reference!
                    vA1 -= delta(vA, vrj, vri - vrj);
                }
                vrs[i] -= delta(vA, vri, std::move(vA1));  // Gauss-Seidel fashion
                return tol_i;
            }));
        }
        for (auto&& result : results) {
            tol = std::max(tol, result.get());
        }

        // fmt::print("tol: {}\n", tol);
        if (tol < options.tol) {
            found = true;
            break;
        }
    }
    return {niter, found};
}

// auto find_rootq(const vec2& r) {
//     auto hb = b / 2.;
//     auto d = hb * hb - c;
//     if (d < 0.) {
//         auto x1 = -hb + (sqrt(-d) if (hb < 0. else -sqrt(-d))*1j;
//     }
//     else {
//         auto x1 = -hb + (sqrt(d) if (hb < 0. else -sqrt(d));
//     }
//     auto x2 = c / x1;
//     return x1, x2;
// }
