#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"
#include "utils/error.h"

#include "metrics/minkowski.h"

#include "kernels/fields_bcs.hpp"

#include <Kokkos_Core.hpp>

#include <iostream>
#include <stdexcept>
#include <string>

using namespace ntt;
using namespace kernel::bc;
using namespace metric;

void errorIf(bool condition, const std::string& message) {
  if (condition) {
    throw std::runtime_error(message);
  }
}

template <Dimension D>
struct DummyFieldsBCs {
  DummyFieldsBCs() {}

  Inline auto ex1(const coord_t<D>&) const -> real_t {
    return TWO;
  }

  Inline auto ex2(const coord_t<D>&) const -> real_t {
    return THREE;
  }

  Inline auto bx2(const coord_t<D>&) const -> real_t {
    return FOUR;
  }

  Inline auto bx3(const coord_t<D>&) const -> real_t {
    return FIVE;
  }
};

Inline auto equal(real_t a, real_t b, const char* msg, real_t acc) -> bool {
  if (not(math::abs(a - b) < acc)) {
    printf("%.12e != %.12e [%.12e] %s\n", a, b, math::abs(a - b), msg);
    return false;
  }
  return true;
}

template <Dimension D>
void testFldsBCs(const std::vector<std::size_t>& res) {
  errorIf(res.size() != (unsigned short)D, "res.size() != D");
  boundaries_t<real_t> sx;
  for (const auto& r : res) {
    sx.emplace_back(ZERO, r);
  }
  const auto      metric = Minkowski<D> { res, sx };
  auto            fset   = DummyFieldsBCs<D> {};
  ndfield_t<D, 6> flds;
  if constexpr (D == Dim::_1D) {
    flds = ndfield_t<D, 6> { "flds", res[0] + 2 * N_GHOSTS };
  } else if constexpr (D == Dim::_2D) {
    flds = ndfield_t<D, 6> { "flds", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS };
  } else if constexpr (D == Dim::_3D) {
    flds = ndfield_t<D, 6> { "flds",
                             res[0] + 2 * N_GHOSTS,
                             res[1] + 2 * N_GHOSTS,
                             res[2] + 2 * N_GHOSTS };
  }

  range_t<D> range;

  if constexpr (D == Dim::_1D) {
    range = CreateRangePolicy<D>({ res[0] / 2 + N_GHOSTS },
                                 { res[0] + 2 * N_GHOSTS });
  } else if constexpr (D == Dim::_2D) {
    range = CreateRangePolicy<D>({ res[0] / 2 + N_GHOSTS, 0 },
                                 { res[0] + 2 * N_GHOSTS, res[1] + N_GHOSTS });
  } else if constexpr (D == Dim::_3D) {
    range = CreateRangePolicy<D>(
      { res[0] / 2 + N_GHOSTS, 0, 0 },
      { res[0] + 2 * N_GHOSTS, res[1] + N_GHOSTS, res[2] + N_GHOSTS });
  }

  const auto xg_edge = (real_t)(sx[0].second);
  const auto dx_abs  = (real_t)(res[0] / 10.0);

  Kokkos::parallel_for(
    "MatchBoundaries_kernel",
    range,
    MatchBoundaries_kernel<SimEngine::SRPIC, decltype(fset), decltype(metric), in::x1>(
      flds,
      fset,
      metric,
      xg_edge,
      dx_abs,
      BC::E | BC::B));

  if constexpr (D == Dim::_1D) {
    Kokkos::parallel_for(
      "MatchBoundaries_kernel",
      CreateRangePolicy<Dim::_1D>({ N_GHOSTS }, { res[0] + N_GHOSTS }),
      Lambda(index_t i1) {
        const auto x       = static_cast<real_t>(i1 - N_GHOSTS);
        const auto factor1 = math::tanh(
          FOUR * math::abs(x + HALF - xg_edge) / dx_abs);
        const auto factor2 = math::tanh(FOUR * math::abs(x - xg_edge) / dx_abs);
        if (not cmp::AlmostEqual(flds(i1, em::ex1), TWO * (ONE - factor1))) {
          printf("%f != %f\n", flds(i1, em::ex1), TWO * (ONE - factor1));
          raise::KernelError(HERE, "incorrect ex1");
        }
        if (not cmp::AlmostEqual(flds(i1, em::ex2), THREE * (ONE - factor2))) {
          printf("%f != %f\n", flds(i1, em::ex2), THREE * (ONE - factor2));
          raise::KernelError(HERE, "incorrect ex2");
        }
        if (not cmp::AlmostEqual(flds(i1, em::bx2), FOUR * (ONE - factor1))) {
          printf("%f != %f\n", flds(i1, em::bx2), FOUR * (ONE - factor1));
          raise::KernelError(HERE, "incorrect bx2");
        }
        if (not cmp::AlmostEqual(flds(i1, em::bx3), FIVE * (ONE - factor1))) {
          printf("%f != %f\n", flds(i1, em::bx3), FIVE * (ONE - factor1));
          raise::KernelError(HERE, "incorrect bx3");
        }
      });
  } else if constexpr (D == Dim::_2D) {
    Kokkos::parallel_for(
      "MatchBoundaries_kernel",
      CreateRangePolicy<Dim::_2D>({ N_GHOSTS, N_GHOSTS },
                                  { res[0] + N_GHOSTS, res[1] + N_GHOSTS }),
      Lambda(index_t i1, index_t i2) {
        const auto x       = static_cast<real_t>(i1 - N_GHOSTS);
        const auto factor1 = math::tanh(
          FOUR * math::abs(x + HALF - xg_edge) / dx_abs);
        const auto factor2 = math::tanh(FOUR * math::abs(x - xg_edge) / dx_abs);
        if (not cmp::AlmostEqual(flds(i1, i2, em::ex1), TWO * (ONE - factor1))) {
          printf("%f != %f\n", flds(i1, i2, em::ex1), TWO * (ONE - factor1));
          raise::KernelError(HERE, "incorrect ex1");
        }
        if (not cmp::AlmostEqual(flds(i1, i2, em::ex2), THREE * (ONE - factor2))) {
          printf("%f != %f\n", flds(i1, i2, em::ex2), THREE * (ONE - factor2));
          raise::KernelError(HERE, "incorrect ex2");
        }
        if (not cmp::AlmostEqual(flds(i1, i2, em::bx2), FOUR * (ONE - factor1))) {
          printf("%f != %f\n", flds(i1, i2, em::bx2), FOUR * (ONE - factor1));
          raise::KernelError(HERE, "incorrect bx2");
        }
        if (not cmp::AlmostEqual(flds(i1, i2, em::bx3), FIVE * (ONE - factor1))) {
          printf("%f != %f\n", flds(i1, i2, em::bx3), FIVE * (ONE - factor1));
          raise::KernelError(HERE, "incorrect bx3");
        }
      });
  } else if constexpr (D == Dim::_3D) {
    Kokkos::parallel_for(
      "MatchBoundaries_kernel",
      CreateRangePolicy<Dim::_3D>(
        { N_GHOSTS, N_GHOSTS, N_GHOSTS },
        { res[0] + N_GHOSTS, res[1] + N_GHOSTS, res[2] + N_GHOSTS }),
      Lambda(index_t i1, index_t i2, index_t i3) {
        const auto x       = static_cast<real_t>(i1 - N_GHOSTS);
        const auto factor1 = math::tanh(
          FOUR * math::abs(x + HALF - xg_edge) / dx_abs);
        const auto factor2 = math::tanh(FOUR * math::abs(x - xg_edge) / dx_abs);
        if (not cmp::AlmostEqual(flds(i1, i2, i3, em::ex1), TWO * (ONE - factor1))) {
          printf("%f != %f\n", flds(i1, i2, i3, em::ex1), TWO * (ONE - factor1));
          raise::KernelError(HERE, "incorrect ex1");
        }
        if (not cmp::AlmostEqual(flds(i1, i2, i3, em::ex2),
                                 THREE * (ONE - factor2))) {
          printf("%f != %f\n", flds(i1, i2, i3, em::ex2), THREE * (ONE - factor2));
          raise::KernelError(HERE, "incorrect ex2");
        }
        if (not cmp::AlmostEqual(flds(i1, i2, i3, em::bx2),
                                 FOUR * (ONE - factor1))) {
          printf("%f != %f\n", flds(i1, i2, i3, em::bx2), FOUR * (ONE - factor1));
          raise::KernelError(HERE, "incorrect bx2");
        }
        if (not cmp::AlmostEqual(flds(i1, i2, i3, em::bx3),
                                 FIVE * (ONE - factor1))) {
          printf("%f != %f\n", flds(i1, i2, i3, em::bx3), FIVE * (ONE - factor1));
          raise::KernelError(HERE, "incorrect bx3");
        }
      });
  }
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    using namespace ntt;

    testFldsBCs<Dim::_1D>({ 24 });
    testFldsBCs<Dim::_2D>({ 64, 32 });
    testFldsBCs<Dim::_3D>({ 14, 22, 15 });

  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}
