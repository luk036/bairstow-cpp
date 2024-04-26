// import numpy as np
// -*- coding: utf-8 -*-
#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#include <doctest/doctest.h>  // for ResultBuilder, CHECK, TEST_CASE

#include <bairstow/rootfinding.hpp>  // for horner, initial_guess, pbairstow...
#include <utility>                   // for pair
#include <vector>                    // for vector

#include "bairstow/vector2.hpp"  // for Vector2
#include "fmt/format.h"          // for print

using namespace numeric;

TEST_CASE("test delta1()") {
    const auto vri = Vector2(-2.0, 0.0);
    const auto vrj = Vector2(4.0, 5.0);
    const auto vrk = Vector2(3.0, 7.0);
    const auto vpj = vri - vrj;
    const auto vpk = vri - vrk;

    auto vA = Vector2(3.0, 3.0);
    vA = delta(vA, vri, vpj);
    auto dr1 = delta(vA, vri, vpk);

    vA = Vector2(3.0, 3.0);
    vA = delta(vA, vri, vpk);
    auto dr2 = delta(vA, vri, vpj);
    CHECK_EQ(dr1.dot(dr1), doctest::Approx(dr2.dot(dr2)));
}

TEST_CASE("test suppress 1") {
    const auto vri = Vector2(-2.0, 0.0);
    const auto vrj = Vector2(4.0, 5.0);
    const auto vrk = Vector2(3.0, 7.0);

    auto vA = Vector2(3.0, 3.0);
    auto vA1 = Vector2(1.0, 2.0);

    suppress(vA, vA1, vri, vrj);
    suppress(vA, vA1, vri, vrk);
    auto dr1 = vA;

    vA = Vector2(3.0, 3.0);
    vA1 = Vector2(1.0, 2.0);
    suppress(vA, vA1, vri, vrk);
    suppress(vA, vA1, vri, vrj);
    auto dr2 = vA;
    CHECK_EQ(dr1.dot(dr1), doctest::Approx(dr2.dot(dr2)));
}

TEST_CASE("test suppress 2") {
    const auto vri = Vector2(-2.0, 0.0);
    const auto vrj = Vector2(4.0, 5.0);
    const auto vrk = Vector2(3.0, 7.0);
    const auto vrl = Vector2(3.0, 7.0);

    auto vA = Vector2(3.0, 3.0);
    auto vA1 = Vector2(1.0, 2.0);

    suppress(vA, vA1, vri, vrj);
    suppress(vA, vA1, vri, vrk);
    suppress(vA, vA1, vri, vrl);
    auto dr1 = vA;

    vA = Vector2(3.0, 3.0);
    vA1 = Vector2(1.0, 2.0);
    suppress(vA, vA1, vri, vrl);
    suppress(vA, vA1, vri, vrk);
    suppress(vA, vA1, vri, vrj);
    auto dr2 = vA;
    CHECK_EQ(dr1.dot(dr1), doctest::Approx(dr2.dot(dr2)));
}

TEST_CASE("test suppress 3") {
    const auto vri = Vector2(-2.0, 0.0);
    const auto vrj = Vector2(4.0, 5.0);
    const auto vrk = Vector2(3.0, 7.0);

    auto vA = Vector2(3.0, 3.0);
    auto vA1 = Vector2(1.0, 2.0);

    suppress2(vA, vA1, vri, vrj);
    suppress2(vA, vA1, vri, vrk);
    auto dr1 = vA;

    vA = Vector2(3.0, 3.0);
    vA1 = Vector2(1.0, 2.0);
    suppress2(vA, vA1, vri, vrk);
    suppress2(vA, vA1, vri, vrj);
    auto dr2 = vA;
    CHECK_EQ(dr1.dot(dr1), doctest::Approx(dr2.dot(dr2)));
}

TEST_CASE("test suppress 4") {
    const auto vri = Vector2(-2.0, 0.0);
    const auto vrj = Vector2(4.0, 5.0);
    const auto vrk = Vector2(3.0, 7.0);
    const auto vrl = Vector2(3.0, 7.0);

    auto vA = Vector2(3.0, 3.0);
    auto vA1 = Vector2(1.0, 2.0);

    suppress2(vA, vA1, vri, vrj);
    suppress2(vA, vA1, vri, vrk);
    suppress2(vA, vA1, vri, vrl);
    auto dr1 = vA;

    vA = Vector2(3.0, 3.0);
    vA1 = Vector2(1.0, 2.0);
    suppress2(vA, vA1, vri, vrl);
    suppress2(vA, vA1, vri, vrk);
    suppress2(vA, vA1, vri, vrj);
    auto dr2 = vA;
    CHECK_EQ(dr1.dot(dr1), doctest::Approx(dr2.dot(dr2)));
}

TEST_CASE("test suppress 5") {
    const auto vri = Vector2(-2.0, 0.0);
    const auto vrj = Vector2(4.0, 5.0);
    // const auto vrk = Vector2(3.0, 7.0);

    auto vA = Vector2(3.0, 3.0);
    auto vA1 = Vector2(1.0, 2.0);

    suppress(vA, vA1, vri, vrj);
    // suppress(vA, vA1, vri, vrk);
    auto dr1 = delta(vA, vri, vA1);

    vA = Vector2(3.0, 3.0);
    vA1 = Vector2(1.0, 2.0);
    suppress2(vA, vA1, vri, vrj);
    //suppress2(vA, vA1, vri, vrk);
    auto dr2 = delta(vA, vri, vA1);
    CHECK_EQ(dr1.dot(dr1), doctest::Approx(dr2.dot(dr2)));
}

TEST_CASE("test horner_eval") {
    auto h = std::vector<double>{1.0, 2.0, 3.0, 4.0};
    auto vA = horner_eval(h, 3, 1.0);
    CHECK(vA == 10.0);
    auto vA1 = horner_eval(h, 2, 1.0);
    CHECK(vA1 == 6.0);
}

TEST_CASE("test horner") {
    auto h = std::vector<double>{1.0, 2.0, 3.0, 4.0};
    auto v = numeric::Vector2(1.0, 2.0);
    auto vA = horner(h, 3, v);
    CHECK(vA.x() == 8.0);
    CHECK(vA.y() == 10.0);
}

TEST_CASE("test initial_guess") {
    auto h = std::vector<double>{5., 2., 9., 6., 2.};
    auto vrs = initial_guess(h);
    CHECK(vrs.size() == 2);
}

TEST_CASE("test root-finding 1") {
    auto h = std::vector<double>{5., 2., 9., 6., 2.};
    auto vrs = initial_guess(h);
    // fmt::print(vrs);
    fmt::print("vrs[1]: {}, {}\n", vrs[1].x(), vrs[1].y());
    auto coeffs1 = h;
    auto N = coeffs1.size() - 1;
    auto vAh = horner(coeffs1, N, vrs[1]);
    fmt::print("{}, {}\n", vAh.x(), vAh.y());
    // fmt::print(coeffs1);
    auto vA1h = horner(coeffs1, N - 2, vrs[1]);
    fmt::print("{}, {}\n", vA1h.x(), vA1h.y());

    auto result = pbairstow_even(h, vrs, Options());
    auto niter = result.first;
    auto found = result.second;
    fmt::print("{}, {}\n", niter, found);

    REQUIRE(found);
    CHECK(niter <= 11);

    // fmt::print([find_rootq(-r[0], -r[1]) for r : vrs]);
}

TEST_CASE("test root-finding 2") {
    auto h = std::vector<double>{10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0};
    auto vrs = initial_guess(h);
    // fmt::print(vrs);
    fmt::print("vrs[1]: {}, {}\n", vrs[1].x(), vrs[1].y());
    auto coeffs1 = h;
    auto N = coeffs1.size() - 1;
    auto vAh = horner(coeffs1, N, vrs[1]);
    fmt::print("{}, {}\n", vAh.x(), vAh.y());
    // fmt::print(coeffs1);
    auto vA1h = horner(coeffs1, N - 2, vrs[1]);
    fmt::print("{}, {}\n", vA1h.x(), vA1h.y());

    auto options = Options();
    options.tolerance = 1e-12;
    auto result = pbairstow_even(h, vrs, options);
    auto niter = result.first;
    auto found = result.second;
    fmt::print("{}, {}\n", niter, found);

    REQUIRE(found);
    CHECK(niter <= 13);
    // fmt::print([find_rootq(-r[0], -r[1]) for r : vrs]);
}

TEST_CASE("test root-finding FIR") {
    // auto vA = vec2{0.1, 1.2};
    // auto vA1 = vec2{2.3, 3.4};
    // auto vr = vec2{4.5, 5.6};
    // auto vrj = vec2{6.7, 7.8};
    // auto vA1 = suppress(vA, vA1, vr, vrj);
    // fmt::print(check_newton(vA, vA1, vr));
    auto h = std::vector<double>{10.0, 34.0, 75.0, 94.0, 150.0, 94.0, 75.0, 34.0, 10.0};
    auto vrs = initial_guess(h);
    // fmt::print(vrs);
    fmt::print("vrs[1]: {}, {}\n", vrs[1].x(), vrs[1].y());
    auto coeffs1 = h;
    auto N = coeffs1.size() - 1;
    auto vAh = horner(coeffs1, N, vrs[1]);
    fmt::print("{}, {}\n", vAh.x(), vAh.y());
    // fmt::print(coeffs1);
    auto vA1h = horner(coeffs1, N - 2, vrs[1]);
    fmt::print("{}, {}\n", vA1h.x(), vA1h.y());

    auto options = Options();
    options.tolerance = 1e-12;
    auto result = pbairstow_even(h, vrs, options);
    auto niter = result.first;
    auto found = result.second;
    fmt::print("{}, {}\n", niter, found);

    CHECK(niter <= 14);
    // fmt::print([find_rootq(-r[0], -r[1]) for r : vrs]);
}
