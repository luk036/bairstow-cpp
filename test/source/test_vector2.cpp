// import numpy as np
// -*- coding: utf-8 -*-
#include <doctest/doctest.h>  // for ResultBuilder, CHECK, TEST_CASE
#include "bairstow/vector2.hpp"  // for Vector2
#include "fmt/format.h"          // for print
#include <vector>

TEST_CASE("test vector2") {
    auto h = std::vector<double>{1.0, 2.0, 3.0, 4.0};
    auto x = 1.0;
    auto y = 2.0;
    auto v = numeric::Vector2<double&, double&>(x, y);
    CHECK(v.x() == 1.0);
    CHECK(v.y() == 2.0);

    auto v2 = numeric::Vector2<double&, double&>(x, y);
    v2 *= 2.0;
    CHECK(v.y() == 4.0);

    // auto vA = horner(h, 3, v);
    // CHECK(vA.x() == 8.0);
    // CHECK(vA.y() == 10.0);
}

