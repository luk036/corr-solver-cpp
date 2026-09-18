#include <corrsolver/version.h>

#include <cassert>
#include <cmath>
#include <corrsolver/linalg.hpp>

auto main() -> int {
    assert(CORRSOLVER_VERSION_MAJOR == 1);

    Arr a(2, 2);
    a(0, 0) = 1.0;
    a(0, 1) = 2.0;
    a(1, 0) = 3.0;
    a(1, 1) = 4.0;

    const auto t = transpose(a);
    assert(t(1, 0) == 2.0);

    const auto p = matmul(a, t);
    assert(p(0, 0) == 5.0);

    return 0;
}
