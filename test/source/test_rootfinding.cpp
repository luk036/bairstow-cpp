// import numpy as np
// -*- coding: utf-8 -*-
#include <doctest/doctest.h>  // for ResultBuilder, CHECK, TEST_CASE

#include <bairstow/rootfinding.hpp>  // for horner, initial_guess, pbairstow...
#include <utility>                   // for pair
#include <vector>                    // for vector

#include "bairstow/vector2.hpp"  // for Vector2
#include "fmt/format.h"          // for print

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
    auto pb = h;
    auto N = pb.size() - 1;
    auto vAh = horner(pb, N, vrs[1]);
    fmt::print("{}, {}\n", vAh.x(), vAh.y());
    // fmt::print(pb);
    auto vA1h = horner(pb, N - 2, vrs[1]);
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
    auto pb = h;
    auto N = pb.size() - 1;
    auto vAh = horner(pb, N, vrs[1]);
    fmt::print("{}, {}\n", vAh.x(), vAh.y());
    // fmt::print(pb);
    auto vA1h = horner(pb, N - 2, vrs[1]);
    fmt::print("{}, {}\n", vA1h.x(), vA1h.y());

    auto options = Options();
    options.tol = 1e-12;
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
    auto pb = h;
    auto N = pb.size() - 1;
    auto vAh = horner(pb, N, vrs[1]);
    fmt::print("{}, {}\n", vAh.x(), vAh.y());
    // fmt::print(pb);
    auto vA1h = horner(pb, N - 2, vrs[1]);
    fmt::print("{}, {}\n", vA1h.x(), vA1h.y());

    auto options = Options();
    options.tol = 1e-12;
    auto result = pbairstow_even(h, vrs, options);
    auto niter = result.first;
    auto found = result.second;
    fmt::print("{}, {}\n", niter, found);

    CHECK(niter <= 14);
    // fmt::print([find_rootq(-r[0], -r[1]) for r : vrs]);
}
